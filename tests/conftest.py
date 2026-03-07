"""Shared test fixtures."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

SAMPLE_GNUCASH_XML = """\
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
<book:id type="guid">book001</book:id>
<gnc:count-data cd:type="account">8</gnc:count-data>
<gnc:count-data cd:type="transaction">8</gnc:count-data>

<gnc:commodity version="2.0.0">
  <cmdty:space>CURRENCY</cmdty:space>
  <cmdty:id>GBP</cmdty:id>
</gnc:commodity>

<!-- ROOT ACCOUNT -->
<gnc:account version="2.0.0">
  <act:name>Root Account</act:name>
  <act:id type="guid">root01</act:id>
  <act:type>ROOT</act:type>
</gnc:account>

<!-- ASSET: Current Account -->
<gnc:account version="2.0.0">
  <act:name>Current Account</act:name>
  <act:id type="guid">acct_current</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- ASSET: Savings Account -->
<gnc:account version="2.0.0">
  <act:name>Savings Account</act:name>
  <act:id type="guid">acct_savings</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- EXPENSE: Expenses:Food -->
<gnc:account version="2.0.0">
  <act:name>Expenses</act:name>
  <act:id type="guid">acct_expenses</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<gnc:account version="2.0.0">
  <act:name>Food</act:name>
  <act:id type="guid">acct_food</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">acct_expenses</act:parent>
</gnc:account>

<!-- TARGET: Unspecified -->
<gnc:account version="2.0.0">
  <act:name>Unspecified</act:name>
  <act:id type="guid">acct_unspec</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- TARGET: Imbalance-GBP -->
<gnc:account version="2.0.0">
  <act:name>Imbalance-GBP</act:name>
  <act:id type="guid">acct_imbalance</act:id>
  <act:type>EXPENSE</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- EUR Account (for non-GBP test) -->
<gnc:account version="2.0.0">
  <act:name>Euro Account</act:name>
  <act:id type="guid">acct_eur</act:id>
  <act:type>BANK</act:type>
  <act:commodity><cmdty:space>CURRENCY</cmdty:space><cmdty:id>EUR</cmdty:id></act:commodity>
  <act:parent type="guid">root01</act:parent>
</gnc:account>

<!-- TX1: Normal categorised (Food) - NOT a candidate -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_normal_food</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-01-10 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Tesco Groceries</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_nf1</split:id><split:memo></split:memo>
      <split:value>-1500/100</split:value><split:quantity>-1500/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_nf2</split:id><split:memo></split:memo>
      <split:value>1500/100</split:value><split:quantity>1500/100</split:quantity>
      <split:account type="guid">acct_food</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX2: Unspecified target (candidate) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_unspec1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-01-15 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Card Payment</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_u1a</split:id><split:memo></split:memo>
      <split:value>-2500/100</split:value><split:quantity>-2500/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_u1b</split:id><split:memo></split:memo>
      <split:value>2500/100</split:value><split:quantity>2500/100</split:quantity>
      <split:account type="guid">acct_unspec</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX3: Another Unspecified target (candidate) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_unspec2</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-02-01 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Direct Debit</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_u2a</split:id><split:memo></split:memo>
      <split:value>-950/100</split:value><split:quantity>-950/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_u2b</split:id><split:memo></split:memo>
      <split:value>950/100</split:value><split:quantity>950/100</split:quantity>
      <split:account type="guid">acct_unspec</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX4: Imbalance-GBP target (candidate) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_imbalance1</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-01-20 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>POS Transaction</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_i1a</split:id><split:memo></split:memo>
      <split:value>-3200/100</split:value><split:quantity>-3200/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_i1b</split:id><split:memo></split:memo>
      <split:value>3200/100</split:value><split:quantity>3200/100</split:quantity>
      <split:account type="guid">acct_imbalance</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX5: NON-GBP (should be excluded) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_eur</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>EUR</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-01-18 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Euro Purchase</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_eur1</split:id><split:memo></split:memo>
      <split:value>-5000/100</split:value><split:quantity>-5000/100</split:quantity>
      <split:account type="guid">acct_eur</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_eur2</split:id><split:memo></split:memo>
      <split:value>5000/100</split:value><split:quantity>5000/100</split:quantity>
      <split:account type="guid">acct_unspec</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX6: FUTURE dated (should be excluded) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_future</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2099-12-31 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Future Payment</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_f1</split:id><split:memo></split:memo>
      <split:value>-100/100</split:value><split:quantity>-100/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_f2</split:id><split:memo></split:memo>
      <split:value>100/100</split:value><split:quantity>100/100</split:quantity>
      <split:account type="guid">acct_unspec</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX7: TRANSFER between own accounts (should be excluded) -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_transfer</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-01-25 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Transfer to Savings</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_t1</split:id><split:memo></split:memo>
      <split:value>-50000/100</split:value><split:quantity>-50000/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_t2</split:id><split:memo></split:memo>
      <split:value>50000/100</split:value><split:quantity>50000/100</split:quantity>
      <split:account type="guid">acct_savings</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

<!-- TX8: Second normal categorised - for history training -->
<gnc:transaction version="2.0.0">
  <trn:id type="guid">tx_normal_food2</trn:id>
  <trn:currency><cmdty:space>CURRENCY</cmdty:space><cmdty:id>GBP</cmdty:id></trn:currency>
  <trn:date-posted><ts:date>2025-02-05 00:00:00 +0000</ts:date></trn:date-posted>
  <trn:description>Sainsburys Weekly Shop</trn:description>
  <trn:splits>
    <trn:split>
      <split:id type="guid">sp_nf2a</split:id><split:memo></split:memo>
      <split:value>-4275/100</split:value><split:quantity>-4275/100</split:quantity>
      <split:account type="guid">acct_current</split:account>
    </trn:split>
    <trn:split>
      <split:id type="guid">sp_nf2b</split:id><split:memo></split:memo>
      <split:value>4275/100</split:value><split:quantity>4275/100</split:quantity>
      <split:account type="guid">acct_food</split:account>
    </trn:split>
  </trn:splits>
</gnc:transaction>

</gnc:book>
</gnc-v2>
"""


@pytest.fixture()
def sample_gnucash_path(tmp_path: Path) -> Path:
    """Create a gzip-compressed GnuCash XML fixture and return its path."""
    path = tmp_path / "test_book.gnucash"
    with gzip.open(path, "wb") as f:
        f.write(SAMPLE_GNUCASH_XML.encode("utf-8"))
    return path


@pytest.fixture()
def sample_gnucash_xml_path(tmp_path: Path) -> Path:
    """Create an uncompressed GnuCash XML fixture."""
    path = tmp_path / "test_book_plain.gnucash"
    path.write_text(SAMPLE_GNUCASH_XML, encoding="utf-8")
    return path
