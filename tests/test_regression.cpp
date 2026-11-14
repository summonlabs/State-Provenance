// ---------------------------------------------------------------------------
// State Provenance - regression tests for audited defects.
// Fenced publish atomic with authority changes, refused restores leave the
// store unchanged and malformed references are refused, registered and
// superseded worker boots survive persistence, a superseded boot never becomes
// current again, format-1 snapshots still decode, and reuse evaluates the
// requested dependency generation.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <thread>
#include <utility>
#include <vector>

#include "stateprovenance/store.hpp"
#include "stateprovenance/persist.hpp"

#include "framework.hpp"

using namespace stateprovenance;

namespace {

RecordData reg_record(StateId sid, ProvenanceId pid) {
    RecordData d;
    d.provenance_id = pid; d.provenance_generation = ProvenanceGeneration(1);
    d.subject_id = sid; d.subject_kind = SubjectKind::TensorState;
    d.state_generation = StateGeneration(1);
    d.producer_id = ProducerId(0xA); d.producer_generation = ProducerGeneration(1);
    d.execution_id = ExecutionId(0xE1); d.attempt_id = AttemptId(0x1); d.attempt_generation = AttemptGeneration(1);
    d.model_id = ModelId(0x7); d.model_generation = ModelGeneration(1);
    d.evidence = EvidenceClass::MEASURED; d.validity = ValidityState::VALID;
    return d;
}

AuthorityClaim claim_for(std::uint64_t epoch, WorkerId w, WorkerBootId b) {
    AuthorityClaim c;
    c.coordinator_epoch = epoch; c.worker_id = w; c.worker_boot = b;
    return c;
}

std::size_t count_with_epoch(const StoreSnapshot& s, std::uint64_t epoch) {
    std::size_t n = 0;
    for (const auto& r : s.records)
        if (r->d.coordinator_epoch == epoch) ++n;
    return n;
}

} // namespace

TEST_CASE(regression_fenced_publish_is_atomic_with_epoch_roll) {
    // Once roll_epoch() has returned, no record whose claim carried the old
    // epoch may still be accepted.
    ProvenanceStore store;
    const WorkerId w(0x51); const WorkerBootId b(0x52);
    store.register_boot(w, b);
    constexpr int kPublishers = 4, kPerPublisher = 2000;
    std::atomic<bool> rolling{false};
    std::atomic<int> publishers_done{0};
    std::vector<std::pair<std::uint64_t, std::size_t>> after_roll;  // (old epoch, its records right after the roll)
    std::thread roller([&]() {
        do {
            const std::uint64_t now = store.roll_epoch();
            after_roll.emplace_back(now - 1, count_with_epoch(store.snapshot(), now - 1));
            rolling.store(true);
        } while (publishers_done.load() < kPublishers);
    });
    std::vector<std::thread> publishers;
    for (int t = 0; t < kPublishers; ++t) {
        publishers.emplace_back([&, t]() {
            while (!rolling.load()) std::this_thread::yield();   // publish while the epoch rolls
            for (int i = 0; i < kPerPublisher; ++i) {
                const std::uint64_t e = store.epoch();
                const std::uint64_t id = 0x100000ull * static_cast<std::uint64_t>(t + 1) + static_cast<std::uint64_t>(i);
                RecordData d = reg_record(StateId(id), ProvenanceId(id));
                d.coordinator_epoch = e;
                (void)store.publish(d, claim_for(e, w, b));
            }
            publishers_done.fetch_add(1);
        });
    }
    for (auto& t : publishers) t.join();
    roller.join();
    std::map<std::uint64_t, std::size_t> per_epoch;
    for (const auto& r : store.snapshot().records) ++per_epoch[r->d.coordinator_epoch];
    std::size_t late = 0;
    for (const auto& [epoch, count] : after_roll) {
        auto it = per_epoch.find(epoch);
        if ((it == per_epoch.end() ? 0 : it->second) != count) ++late;
    }
    CHECK(!after_roll.empty());
    CHECK_EQ(late, std::size_t(0));
}

TEST_CASE(regression_refused_restore_leaves_store_unchanged) {
    ProvenanceStore live;
    const WorkerId w(0x61); const WorkerBootId b(0x62);
    live.register_boot(w, b);
    live.roll_epoch(); live.roll_epoch();   // epoch 3
    CHECK(live.publish(reg_record(StateId(1), ProvenanceId(11))).ok);
    RecordData child = reg_record(StateId(2), ProvenanceId(12));
    child.input_states = {StateId(1)};
    CHECK(live.publish(child).ok);

    // An older snapshot (epoch 1) whose two records name each other as input.
    StoreSnapshot bad;
    RecordData x = reg_record(StateId(5), ProvenanceId(15)); x.input_states = {StateId(6)};
    RecordData y = reg_record(StateId(6), ProvenanceId(16)); y.input_states = {StateId(5)};
    bad.records.push_back(std::make_shared<const ProvenanceRecord>(x));
    bad.records.push_back(std::make_shared<const ProvenanceRecord>(y));
    bad.authority.epoch = 1;
    CHECK_THROWS(live.restore(bad));
    CHECK_EQ(live.size(), std::size_t(2));
    CHECK_EQ(live.epoch(), std::uint64_t(3));
    CHECK(!live.evaluate_claim(claim_for(1, w, b)).accepted());
    CHECK(live.evaluate_claim(claim_for(3, w, b)).accepted());
}

TEST_CASE(regression_restore_refuses_malformed_references) {
    ProvenanceStore s;
    CHECK(s.publish(reg_record(StateId(1), ProvenanceId(11))).ok);
    RecordData b = reg_record(StateId(2), ProvenanceId(12));
    b.parent_provenance = {ProvenanceId(11)};
    CHECK(s.publish(b).ok);
    const StoreSnapshot snap = s.snapshot();   // records sorted by subject id

    // The parent record is missing: the child's parent_provenance dangles.
    StoreSnapshot dangling = snap;
    dangling.records.erase(dangling.records.begin());
    dangling.status.erase(StateId(1));
    ProvenanceStore r1;
    CHECK_THROWS(r1.restore(dangling));
    CHECK_EQ(r1.size(), std::size_t(0));

    // A status entry for a subject without a record.
    StoreSnapshot phantom = snap;
    phantom.status[StateId(99)] = RecordStatus{};
    ProvenanceStore r2;
    CHECK_THROWS(r2.restore(phantom));
    CHECK_EQ(r2.size(), std::size_t(0));

    ProvenanceStore r3;
    CHECK_NOTHROW(r3.restore(snap));
    CHECK_EQ(r3.size(), std::size_t(2));
}

TEST_CASE(regression_registered_boot_survives_persistence) {
    // A worker that registered but has not published yet keeps its boot
    // across save/recover.
    ProvenanceStore s;
    const WorkerId w(0x71); const WorkerBootId b(0x72);
    s.register_boot(w, b);
    const StoreSnapshot snap = s.snapshot();
    CHECK(snap.authority.boots.count(w) == 1);
    auto bytes = encode_snapshot(snap);
    auto dec = decode_snapshot(bytes.data(), bytes.size());
    CHECK(dec.snapshot.has_value());
    if (dec.snapshot) {
        ProvenanceStore r;
        r.restore(*dec.snapshot);
        CHECK(r.evaluate_claim(claim_for(r.epoch(), w, b)).accepted());
    }
}

TEST_CASE(regression_superseded_boot_never_becomes_current_again) {
    const WorkerId w(0x81); const WorkerBootId b1(0x82), b2(0x83);
    AuthorityState a;
    a.register_boot(w, b1);
    a.register_boot(w, b2);   // the worker restarted
    a.register_boot(w, b1);   // its old incarnation registers again
    CHECK(!a.evaluate_claim(claim_for(1, w, b1)).accepted());
    CHECK(a.evaluate_claim(claim_for(1, w, b2)).accepted());

    // Also after the store is persisted and recovered.
    ProvenanceStore s;
    s.register_boot(w, b1);
    s.register_boot(w, b2);
    auto bytes = encode_snapshot(s.snapshot());
    auto dec = decode_snapshot(bytes.data(), bytes.size());
    CHECK(dec.snapshot.has_value());
    if (dec.snapshot) {
        ProvenanceStore r;
        r.restore(*dec.snapshot);
        r.register_boot(w, b1);
        CHECK(!r.evaluate_claim(claim_for(r.epoch(), w, b1)).accepted());
        CHECK(r.evaluate_claim(claim_for(r.epoch(), w, b2)).accepted());
    }
}

TEST_CASE(regression_format1_snapshot_still_decodes) {
    // A payload in format 1 (no superseded-boot section) still decodes.
    RecordData d = reg_record(StateId(1), ProvenanceId(11));
    AuthoritySnapshot auth;
    auth.epoch = 4;
    auth.boots[WorkerId(0x91)] = WorkerBootId(0x92);
    BinaryWriter w;
    w.u64(kPayloadMagic);
    w.u32(1);
    w.u64(1);
    write_record(w, d);
    w.u32(1);
    w.u64(StateId(1).get());
    w.u8(static_cast<std::uint8_t>(ValidityState::VALID));
    w.u8(static_cast<std::uint8_t>(InvalidationReason::Unknown));
    w.u64(1);
    w.str("");
    write_authority(w, auth);
    auto dec = decode_snapshot(w.data().data(), w.data().size());
    CHECK(dec.error.empty());
    CHECK(dec.snapshot.has_value());
    if (dec.snapshot) {
        ProvenanceStore r;
        r.restore(*dec.snapshot);
        CHECK_EQ(r.size(), std::size_t(1));
        CHECK_EQ(r.epoch(), std::uint64_t(4));
        CHECK(r.evaluate_claim(claim_for(4, WorkerId(0x91), WorkerBootId(0x92))).accepted());
    }
}

TEST_CASE(regression_reuse_checks_dependency_generation) {
    RecordData d = reg_record(StateId(1), ProvenanceId(11));
    d.dependencies = { Dependency{DependencyId(0x5), DependencyGeneration(1), "model", "rev1"} };
    const ProvenanceRecord rec(d);
    ReuseRequest same; same.dependency_generation = DependencyGeneration(1);
    CHECK(evaluate_reuse(rec, same).eligibility == ReuseEligibility::ELIGIBLE);
    ReuseRequest rolled; rolled.dependency_generation = DependencyGeneration(2);
    auto dec = evaluate_reuse(rec, rolled);
    CHECK(dec.eligibility == ReuseEligibility::INCOMPATIBLE);
    CHECK(std::find(dec.failed_dimensions.begin(), dec.failed_dimensions.end(),
                    CompatibilityDimension::DependencyGen) != dec.failed_dimensions.end());

    ProvenanceStore s;
    CHECK(s.publish(d).ok);
    CHECK(s.check_reuse(StateId(1), rolled).eligibility == ReuseEligibility::INCOMPATIBLE);
}
