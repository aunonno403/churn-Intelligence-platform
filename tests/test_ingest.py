"""End-to-end smoke test of ingest + labels + cleanup on tiny files shaped like the KKBox data."""

import io
from datetime import date

import duckdb
import py7zr
import pytest

from churn.config import Settings
from churn.data.ingest import build_raw_tables, clean_raw
from churn.data.labels import build_labels, validate_against_official
from churn.data.logs_filter import LineFilter

TX_HEADER = (
    "msno,payment_method_id,payment_plan_days,plan_list_price,actual_amount_paid,"
    "is_auto_renew,transaction_date,membership_expire_date,is_cancel"
)
LOG_HEADER = "msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs"
CSVS = {
    # nested + split across v1/v2 like the extracted archives; one duplicated row across files
    "a/transactions.csv": [
        TX_HEADER,
        "u1,41,30,149,149,1,20170110,20170210,0",
        "u2,41,30,149,149,1,20170110,20170215,0",
    ],
    "b/transactions_v2.csv": [
        TX_HEADER,
        "u1,41,30,149,149,1,20170110,20170210,0",  # duplicate
        "u1,41,30,149,149,1,20170210,20170312,0",
    ],
    "a/members_v3.csv": [
        "msno,city,bd,gender,registered_via,registration_init_time",
        "u1,1,0,,7,20150101",
        "u2,5,28,male,9,20160301",
    ],
    "a/train.csv": ["msno,is_churn", "u1,0", "u2,1"],
}
# The logs stay compressed, as after `churn download`, and are filtered while decompressing.
ARCHIVES = {
    "a/user_logs.csv.7z": [
        LOG_HEADER,
        "u1,20170105,1,0,0,0,10,8,2500.5",
        "u1,20150101,1,0,0,0,10,8,2500.5",  # before logs_start -> dropped
        "stranger,20170105,1,0,0,0,10,8,2500.5",  # no transactions -> dropped
    ],
    "b/user_logs_v2.csv.7z": [LOG_HEADER, "u2,20170301,2,1,0,0,5,6,1200.0"],
}


@pytest.fixture
def built(tmp_path):
    raw = tmp_path / "raw"
    for rel, lines in CSVS.items():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
    for rel, lines in ARCHIVES.items():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(p, "w") as sz:
            sz.writestr("\n".join(lines) + "\n", p.name.removesuffix(".7z"))
    settings = Settings(data_dir=tmp_path, sample_mod=1, sample_keep=1)  # keep everyone
    with duckdb.connect(str(settings.db_path)) as con:
        build_raw_tables(con, settings)
        build_labels(con, settings)
        report = validate_against_official(con, settings)
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ["transactions", "user_logs"]
        }
        gender = con.execute("SELECT gender FROM members WHERE msno = 'u1'").fetchone()[0]
    return settings, report, counts, gender


def test_raw_tables(built):
    _, _, counts, gender = built
    assert counts == {"transactions": 3, "user_logs": 2}  # deduped; logs filtered
    assert gender is None


def test_labels_match_official(built):
    _, report, _, _ = built
    assert report.in_our_cohort == 2
    assert report.agreement == 1.0


def test_clean_raw_keeps_database(built):
    settings = built[0]
    clean_raw(settings)
    assert not settings.raw_dir.exists()
    assert settings.db_path.exists()


def test_clean_raw_refuses_without_database(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(RuntimeError):
        clean_raw(Settings(data_dir=tmp_path))
    assert (tmp_path / "raw").exists()


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_line_filter_is_chunk_boundary_safe(chunk_size):
    data = (LOG_HEADER + "\nu1,20170101,x\nu2,20170101,x\nu1,20100101,x\nu1,20170102,x").encode()
    out = io.BytesIO()
    filt = LineFilter(out, {b"u1"}, date(2016, 1, 1))
    for i in range(0, len(data), chunk_size):
        filt.feed(data[i : i + chunk_size])
    filt.finish()
    assert out.getvalue().decode().splitlines() == [LOG_HEADER, "u1,20170101,x", "u1,20170102,x"]
