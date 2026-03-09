"""System tests with varied synthetic data scenarios.

Each scenario tests a different combination of data shapes that exercise
different code paths in the pipeline.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from PIL import Image, ImageDraw

from gnc_enrich.config import RunConfig
from gnc_enrich.domain.models import ReviewDecision
from gnc_enrich.gnucash.loader import GnuCashLoader
from gnc_enrich.review.service import ReviewQueueService
from gnc_enrich.services.pipeline import EnrichmentPipeline
from gnc_enrich.state.repository import StateRepository
from gnc_enrich.apply.engine import ApplyEngine


def _make_gnucash_gz(path: Path, xml: str) -> None:
    with gzip.open(path, "wb") as f:
        f.write(xml.encode("utf-8"))


def _make_eml(path: Path, *, subject: str, sender: str, body: str, date: str) -> None:
    path.write_text(
        f"Content-Type: text/plain; charset=\"utf-8\"\n"
        f"Date: {date}\n"
        f"From: {sender}\n"
        f"To: user@example.com\n"
        f"Subject: {subject}\n"
        f"Message-ID: <{path.stem}@test>\n"
        f"MIME-Version: 1.0\n"
        f"\n"
        f"{body}\n",
        encoding="utf-8",
    )


def _make_receipt(path: Path, lines: list[str]) -> None:
    img = Image.new("RGB", (400, 50 * len(lines) + 60), "white")
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 40
    img.save(path)


# ---------------------------------------------------------------------------
# Scenario 1: Only Imbalance-GBP targets, no Unspecified account at all
# ---------------------------------------------------------------------------

IMBALANCE_ONLY_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">imbbook01</book:id>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Current</act:name>
  <act:id type="guid">acct_cur</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Imbalance-GBP</act:name>
  <act:id type="guid">acct_imb</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_imb1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-05-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Shop Purchase</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_i1a</split:id><split:value>-3500/100</split:value><split:account type="guid">acct_cur</split:account></trn:split>
    <trn:split><split:id type="guid">sp_i1b</split:id><split:value>3500/100</split:value><split:account type="guid">acct_imb</split:account></trn:split>
  </trn:splits>
</gnc:transaction>
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_imb2</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-05-10 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Online Order</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_i2a</split:id><split:value>-7999/100</split:value><split:account type="guid">acct_cur</split:account></trn:split>
    <trn:split><split:id type="guid">sp_i2b</split:id><split:value>7999/100</split:value><split:account type="guid">acct_imb</split:account></trn:split>
  </trn:splits>
</gnc:transaction>
</gnc:book>
</gnc-v2>
"""


class TestImbalanceOnlyScenario:
    """When GnuCash file has Imbalance-GBP but no Unspecified account."""

    def test_pipeline_finds_imbalance_candidates(self, tmp_path: Path) -> None:
        gnc = tmp_path / "imb.gnucash"
        _make_gnucash_gz(gnc, IMBALANCE_ONLY_XML)
        state = tmp_path / "state"
        state.mkdir()
        emails = tmp_path / "emails"
        emails.mkdir()
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        processed = tmp_path / "processed"
        processed.mkdir()

        config = RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=receipts,
            processed_receipts_dir=processed, state_dir=state,
        )
        result = EnrichmentPipeline().run(config)
        assert result.proposal_count == 2

    def test_apply_cycle_with_imbalance(self, tmp_path: Path) -> None:
        gnc = tmp_path / "imb.gnucash"
        _make_gnucash_gz(gnc, IMBALANCE_ONLY_XML)
        state = tmp_path / "state"
        state.mkdir()
        emails = tmp_path / "emails"
        emails.mkdir()
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        processed = tmp_path / "processed"
        processed.mkdir()

        config = RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=receipts,
            processed_receipts_dir=processed, state_dir=state,
        )
        EnrichmentPipeline().run(config)

        st = StateRepository(state)
        svc = ReviewQueueService(st)
        for p in svc.all_proposals():
            svc.submit_decision(ReviewDecision(
                tx_id=p.tx_id, action="approve",
                final_description=p.suggested_description,
                final_splits=p.suggested_splits,
            ))

        engine = ApplyEngine()
        engine.apply(state)

        audit = st.load_audit_log()
        assert len(audit) == 2


# ---------------------------------------------------------------------------
# Scenario 2: Zero candidates — all transactions already categorized
# ---------------------------------------------------------------------------

ALL_CATEGORIZED_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">catbook01</book:id>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Current</act:name>
  <act:id type="guid">acct_cur</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Expenses</act:name>
  <act:id type="guid">acct_exp</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Food</act:name>
  <act:id type="guid">acct_food</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">acct_exp</act:parent>
</gnc:account>
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_done1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-04-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Supermarket Shop</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_d1a</split:id><split:value>-2500/100</split:value><split:account type="guid">acct_cur</split:account></trn:split>
    <trn:split><split:id type="guid">sp_d1b</split:id><split:value>2500/100</split:value><split:account type="guid">acct_food</split:account></trn:split>
  </trn:splits>
</gnc:transaction>
</gnc:book>
</gnc-v2>
"""


class TestZeroCandidatesScenario:
    def test_pipeline_with_no_candidates(self, tmp_path: Path) -> None:
        gnc = tmp_path / "done.gnucash"
        _make_gnucash_gz(gnc, ALL_CATEGORIZED_XML)
        state = tmp_path / "state"
        state.mkdir()
        (tmp_path / "emails").mkdir()
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        result = EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=tmp_path / "emails",
            receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))
        assert result.proposal_count == 0

    def test_review_queue_empty(self, tmp_path: Path) -> None:
        gnc = tmp_path / "done.gnucash"
        _make_gnucash_gz(gnc, ALL_CATEGORIZED_XML)
        state = tmp_path / "state"
        state.mkdir()
        (tmp_path / "emails").mkdir()
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=tmp_path / "emails",
            receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))
        svc = ReviewQueueService(StateRepository(state))
        assert svc.total_count == 0
        assert svc.next_proposal() is None


# ---------------------------------------------------------------------------
# Scenario 3: Wider date window changes match coverage
# ---------------------------------------------------------------------------

DATED_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">datebook01</book:id>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Current</act:name>
  <act:id type="guid">acct_cur</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Unspecified</act:name>
  <act:id type="guid">acct_unspec</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>
<!-- Transaction on 2023-07-01 -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_date1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-07-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Water Bill</trn:description>
  <trn:splits>
    <trn:split><split:id type="guid">sp_d1a</split:id><split:value>-3850/100</split:value><split:account type="guid">acct_cur</split:account></trn:split>
    <trn:split><split:id type="guid">sp_d1b</split:id><split:value>3850/100</split:value><split:account type="guid">acct_unspec</split:account></trn:split>
  </trn:splits>
</gnc:transaction>
</gnc:book>
</gnc-v2>
"""


class TestDateWindowScenario:
    """Verify that date_window_days actually affects email matching coverage."""

    def test_narrow_window_fewer_matches(self, tmp_path: Path) -> None:
        gnc = tmp_path / "dated.gnucash"
        _make_gnucash_gz(gnc, DATED_XML)
        state = tmp_path / "state"
        state.mkdir()
        emails = tmp_path / "emails"
        emails.mkdir()
        _make_eml(
            emails / "water_bill.eml",
            subject="Water bill payment of £38.50",
            sender='"Water Co" <billing@waterco.example>',
            body="Your payment of £38.50 has been received. Date: 20 July 2023.",
            date="Thu, 20 Jul 2023 10:00:00 +0000",
        )
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        config_narrow = RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
            date_window_days=3,
        )
        EnrichmentPipeline().run(config_narrow)
        st = StateRepository(state)
        props_narrow = st.load_proposals()
        narrow_emails = props_narrow[0].evidence.emails if props_narrow[0].evidence else []

        state2 = tmp_path / "state2"
        state2.mkdir()
        config_wide = RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state2,
            date_window_days=30,
        )
        EnrichmentPipeline().run(config_wide)
        st2 = StateRepository(state2)
        props_wide = st2.load_proposals()
        wide_emails = props_wide[0].evidence.emails if props_wide[0].evidence else []

        assert len(wide_emails) >= len(narrow_emails)

    def test_amount_tolerance_affects_matching(self, tmp_path: Path) -> None:
        gnc = tmp_path / "dated.gnucash"
        _make_gnucash_gz(gnc, DATED_XML)
        emails = tmp_path / "emails"
        emails.mkdir()
        _make_eml(
            emails / "close_amount.eml",
            subject="Payment of £39.00 received",
            sender='"Utility" <billing@util.example>',
            body="We received £39.00 on 1 July 2023.",
            date="Sat, 01 Jul 2023 12:00:00 +0000",
        )
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        state_tight = tmp_path / "state_tight"
        state_tight.mkdir()
        EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state_tight,
            amount_tolerance=0.10,
        ))

        state_loose = tmp_path / "state_loose"
        state_loose.mkdir()
        EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state_loose,
            amount_tolerance=2.00,
        ))

        props_tight = StateRepository(state_tight).load_proposals()
        props_loose = StateRepository(state_loose).load_proposals()
        tight_emails = props_tight[0].evidence.emails if props_tight[0].evidence else []
        loose_emails = props_loose[0].evidence.emails if props_loose[0].evidence else []
        assert len(loose_emails) >= len(tight_emails)


# ---------------------------------------------------------------------------
# Scenario 4: Multiple receipt images, only one matches
# ---------------------------------------------------------------------------


class TestMultipleReceiptsScenario:
    def test_correct_receipt_matched(self, tmp_path: Path) -> None:
        gnc = tmp_path / "rcpt.gnucash"
        _make_gnucash_gz(gnc, DATED_XML)
        state = tmp_path / "state"
        state.mkdir()
        emails = tmp_path / "emails"
        emails.mkdir()
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        (tmp_path / "processed").mkdir()

        _make_receipt(receipts / "wrong_total.jpg", ["SHOP A", "Total: 12.00"])
        _make_receipt(receipts / "close_total.jpg", ["WATER CO", "Total: 38.50"])
        _make_receipt(receipts / "another_wrong.png", ["CAFE B", "Total: 5.50"])

        result = EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=receipts,
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))
        assert result.proposal_count == 1

        st = StateRepository(state)
        props = st.load_proposals()
        assert len(props) == 1


# ---------------------------------------------------------------------------
# Scenario 5: Deeply nested email subdirectories
# ---------------------------------------------------------------------------


class TestDeepNestedEmailDirs:
    def test_emails_in_nested_subdirs_found(self, tmp_path: Path) -> None:
        gnc = tmp_path / "deep.gnucash"
        _make_gnucash_gz(gnc, DATED_XML)
        state = tmp_path / "state"
        state.mkdir()
        emails = tmp_path / "emails"
        (emails / "level1" / "level2").mkdir(parents=True)
        _make_eml(
            emails / "level1" / "level2" / "deep.eml",
            subject="Payment of £38.50",
            sender='"Deep" <deep@example.com>',
            body="A payment of £38.50 was processed on 1 July 2023.",
            date="Sat, 01 Jul 2023 08:00:00 +0000",
        )
        _make_eml(
            emails / "top.eml",
            subject="Newsletter",
            sender='"News" <news@example.com>',
            body="Weekly newsletter content.",
            date="Mon, 03 Jul 2023 08:00:00 +0000",
        )
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=emails, receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))

        from gnc_enrich.email.index import EmailIndexRepository
        index = EmailIndexRepository()
        index.build_or_load(emails, state)
        assert len(index.entries) == 2


# ---------------------------------------------------------------------------
# Scenario 6: Empty book (no transactions at all)
# ---------------------------------------------------------------------------

EMPTY_BOOK_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">emptybook01</book:id>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
</gnc:book>
</gnc-v2>
"""


class TestEmptyBookScenario:
    def test_pipeline_handles_empty_book(self, tmp_path: Path) -> None:
        gnc = tmp_path / "empty.gnucash"
        _make_gnucash_gz(gnc, EMPTY_BOOK_XML)
        state = tmp_path / "state"
        state.mkdir()
        (tmp_path / "emails").mkdir()
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        result = EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=tmp_path / "emails",
            receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))
        assert result.proposal_count == 0
        assert result.skipped_count == 0


# ---------------------------------------------------------------------------
# Scenario 7: Realistic GnuCash with book:slots (matches real format)
# ---------------------------------------------------------------------------

REALISTIC_FORMAT_XML = """\
<?xml version="1.0" encoding="utf-8" ?>
<gnc-v2
     xmlns:gnc="http://www.gnucash.org/XML/gnc"
     xmlns:act="http://www.gnucash.org/XML/act"
     xmlns:book="http://www.gnucash.org/XML/book"
     xmlns:cd="http://www.gnucash.org/XML/cd"
     xmlns:cmdty="http://www.gnucash.org/XML/cmdty"
     xmlns:slot="http://www.gnucash.org/XML/slot"
     xmlns:split="http://www.gnucash.org/XML/split"
     xmlns:trn="http://www.gnucash.org/XML/trn"
     xmlns:ts="http://www.gnucash.org/XML/ts">
<gnc:count-data cd:type="book">1</gnc:count-data>
<gnc:book version="2.0.0">
<book:id type="guid">a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4</book:id>
<book:slots>
  <slot>
    <slot:key>features</slot:key>
    <slot:value type="frame">
      <slot>
        <slot:key>Account GUID based bayesian with flat KVP</slot:key>
        <slot:value type="string">Requires GnuCash 2.6.19+</slot:value>
      </slot>
    </slot:value>
  </slot>
</book:slots>
<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id>
</gnc:commodity>
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">a000000000000000000000000000root</act:id>
  <act:type>ROOT</act:type>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Assets</act:name>
  <act:id type="guid">a000000000000000000000000assets</act:id>
  <act:type>ASSET</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">a000000000000000000000000000root</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Bank</act:name>
  <act:id type="guid">a00000000000000000000000000bank</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">a000000000000000000000000assets</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Expenses</act:name>
  <act:id type="guid">a0000000000000000000000expenses</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">a000000000000000000000000000root</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Transport</act:name>
  <act:id type="guid">a000000000000000000000transport</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">a0000000000000000000000expenses</act:parent>
</gnc:account>
<gnc:account version="2.0.0">
  <act:name>Unspecified</act:name>
  <act:id type="guid">a00000000000000000000unspecified</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">a000000000000000000000000000root</act:parent>
</gnc:account>
<gnc:transaction version="2.0.0">
  <trn:id type="guid">b00000000000000000000000000tx01</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2023-09-15 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>TRAIN TICKET</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">c00000000000000000000000sp01a</split:id>
      <split:memo></split:memo>
      <split:value>-4250/100</split:value>
      <split:quantity>-4250/100</split:quantity>
      <split:account type="guid">a00000000000000000000000000bank</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">c00000000000000000000000sp01b</split:id>
      <split:memo></split:memo>
      <split:value>4250/100</split:value>
      <split:quantity>4250/100</split:quantity>
      <split:account type="guid">a00000000000000000000unspecified</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>
</gnc:book>
</gnc-v2>
"""


class TestRealisticFormatScenario:
    """GnuCash file with book:slots, proper 32-char hex GUIDs, and nested accounts."""

    def test_pipeline_handles_realistic_format(self, tmp_path: Path) -> None:
        gnc = tmp_path / "realistic.gnucash"
        _make_gnucash_gz(gnc, REALISTIC_FORMAT_XML)
        state = tmp_path / "state"
        state.mkdir()
        (tmp_path / "emails").mkdir()
        (tmp_path / "receipts").mkdir()
        (tmp_path / "processed").mkdir()

        result = EnrichmentPipeline().run(RunConfig(
            gnucash_path=gnc, emails_dir=tmp_path / "emails",
            receipts_dir=tmp_path / "receipts",
            processed_receipts_dir=tmp_path / "processed", state_dir=state,
        ))
        assert result.proposal_count == 1

    def test_loader_reads_slots_without_crashing(self, tmp_path: Path) -> None:
        gnc = tmp_path / "realistic.gnucash"
        _make_gnucash_gz(gnc, REALISTIC_FORMAT_XML)
        loader = GnuCashLoader()
        txs = loader.load_transactions(gnc)
        assert len(txs) == 1
        assert txs[0].description == "TRAIN TICKET"
