"""Read-only audit of WRDS fixed-income data access.

Run from a terminal:
    python economic_shock_sensitivity/wrds_fixed_income_access_audit.py

Or in Jupyter:
    %run economic_shock_sensitivity/wrds_fixed_income_access_audit.py

The script reads only database metadata and at most one row from candidate tables.
It never prints or stores the WRDS password.
"""

from __future__ import annotations

import getpass
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wrds


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "wrds_fixed_income_access_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

LIBRARY_TERMS = (
    "bond", "trace", "fisd", "fixed", "income", "lipper", "factset",
    "treasury", "mfl", "crsp", "wrdsapps", "comp",
)
TABLE_TERMS = (
    "bond", "trace", "fisd", "issue", "rating", "return", "price",
    "yield", "spread", "maturity", "coupon", "fund", "holding", "treasury",
)
CONCEPT_PATTERNS = {
    "bond_identifier": r"cusip|isin|issue_id|bond_id|security_id|secid",
    "investor_identifier": r"fund|portfolio|manager|institution|entity|owner",
    "holding": r"holding|position|par_value|face_value|market_value|shares|quantity",
    "observation_date": r"date|month|quarter|report|filing",
    "return_or_price": r"return|ret$|price|prc|yield",
    "maturity_or_duration": r"maturity|duration|modified_duration|effective_duration",
    "coupon": r"coupon",
    "credit_rating": r"rating|grade|investment_grade",
    "issue_size": r"amount_outstanding|offering_amount|issue_size|principal",
    "liquidity": r"volume|trade_count|bid|ask|turnover|liquidity",
    "issuer_mapping": r"issuer|gvkey|permco|company|entity",
}


def normalized_list(values):
    return sorted({str(value).strip() for value in values if str(value).strip()})


def save_frame(frame: pd.DataFrame, name: str):
    frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    safe = frame.copy()
    for column in safe.select_dtypes(include="object"):
        safe[column] = safe[column].map(lambda value: str(value) if value is not None else None)
    safe.to_parquet(OUTPUT / f"{name}.parquet", index=False)


def connect():
    username = input("WRDS username: ").strip()
    password = getpass.getpass("WRDS password (not displayed or stored): ")
    connection = wrds.Connection(wrds_username=username, wrds_password=password)
    connection.raw_sql("select 1 as connection_test")
    print("WRDS connection established.")
    return connection, username


db, username = connect()

try:
    libraries = normalized_list(db.list_libraries())
    library_frame = pd.DataFrame({"library": libraries})
    library_frame["fixed_income_name_match"] = library_frame.library.str.lower().map(
        lambda value: any(term in value for term in LIBRARY_TERMS)
    )
    save_frame(library_frame, "all_accessible_libraries")

    candidate_libraries = library_frame.loc[library_frame.fixed_income_name_match, "library"].tolist()
    print(f"Accessible libraries: {len(libraries):,}")
    print(f"Candidate libraries: {len(candidate_libraries):,}")

    table_rows = []
    errors = []
    for number, library in enumerate(candidate_libraries, 1):
        try:
            tables = normalized_list(db.list_tables(library=library))
            for table in tables:
                name_match = any(term in table.lower() for term in TABLE_TERMS)
                table_rows.append({"library": library, "table": table, "table_name_match": name_match})
            print(f"[{number:02d}/{len(candidate_libraries):02d}] {library}: {len(tables):,} tables")
        except Exception as error:
            try:
                db.connection.rollback()
            except Exception:
                pass
            errors.append({"stage": "list_tables", "library": library, "table": None,
                           "error_type": type(error).__name__, "message": str(error)[:500]})

    table_inventory = pd.DataFrame(table_rows)
    save_frame(table_inventory, "candidate_library_tables")

    # Describe tables whose names indicate likely fixed-income relevance. Broad
    # CRSP/Compustat libraries are included above, but irrelevant equity tables
    # are not individually probed.
    likely_tables = table_inventory.loc[table_inventory.table_name_match].copy()
    column_rows = []
    access_rows = []
    for number, row in enumerate(likely_tables.itertuples(index=False), 1):
        library, table = row.library, row.table
        qualified = f"{library}.{table}"
        try:
            description = db.describe_table(library=library, table=table).reset_index()
            description.columns = [str(column).lower() for column in description.columns]
            name_column = next((column for column in description.columns
                                if column in {"name", "column_name", "variable"}), None)
            type_column = next((column for column in description.columns
                                if "type" in column), None)
            if name_column is None:
                raise ValueError(f"Could not identify column-name field in {list(description.columns)}")
            for position, (_, values) in enumerate(description.iterrows(), 1):
                column_rows.append({"library": library, "table": table, "ordinal_position": position,
                                    "column_name": str(values[name_column]),
                                    "data_type": str(values[type_column]) if type_column else None})
            # A one-row query distinguishes catalog visibility from actual SELECT access.
            db.raw_sql(f"select * from {qualified} limit 1")
            access_rows.append({"library": library, "table": table, "select_access": True,
                                "columns": len(description), "error_type": None})
        except Exception as error:
            # PostgreSQL leaves a transaction unusable after some catalog or
            # permission errors. Roll it back so one inaccessible table does
            # not make every subsequent probe appear inaccessible.
            try:
                db.connection.rollback()
            except Exception:
                pass
            access_rows.append({"library": library, "table": table, "select_access": False,
                                "columns": None, "error_type": type(error).__name__})
            errors.append({"stage": "describe_or_probe", "library": library, "table": table,
                           "error_type": type(error).__name__, "message": str(error)[:500]})
        if number % 25 == 0 or number == len(likely_tables):
            print(f"Probed {number:,}/{len(likely_tables):,} likely tables")

    columns = pd.DataFrame(column_rows)
    access = pd.DataFrame(access_rows)
    save_frame(columns, "likely_table_columns")
    save_frame(access, "likely_table_access")
    save_frame(pd.DataFrame(errors), "audit_errors")

    concept_rows = []
    if not columns.empty:
        grouped = columns.groupby(["library", "table"], sort=True)
        for (library, table), group in grouped:
            names = " ".join(group.column_name.str.lower())
            record = {"library": library, "table": table}
            for concept, pattern in CONCEPT_PATTERNS.items():
                record[concept] = bool(re.search(pattern, names))
            record["concepts_present"] = sum(record[concept] for concept in CONCEPT_PATTERNS)
            concept_rows.append(record)
    concepts = pd.DataFrame(concept_rows)
    if not concepts.empty:
        concepts = concepts.merge(access[["library", "table", "select_access"]],
                                  on=["library", "table"], how="left", validate="one_to_one")
        concepts = concepts.sort_values(["select_access", "concepts_present"], ascending=False)
    save_frame(concepts, "fixed_income_concept_coverage")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrds_username": username,
        "read_only": True,
        "accessible_libraries": len(libraries),
        "candidate_libraries": len(candidate_libraries),
        "candidate_tables": len(table_inventory),
        "likely_tables_probed": len(likely_tables),
        "selectable_likely_tables": int(access.select_access.sum()) if not access.empty else 0,
        "output_directory": str(OUTPUT),
    }
    (OUTPUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nTop fixed-income candidates")
    if concepts.empty:
        print("No matching table metadata was found.")
    else:
        print(concepts.head(30).to_string(index=False))
    print(f"\nAudit saved to: {OUTPUT}")
finally:
    db.close()
    print("WRDS connection closed.")
