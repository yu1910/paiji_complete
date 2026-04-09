import json
from collections import Counter
from typing import Any
import pandas as pd
from arrange_library.core.preprocessing.base_imbalance_handler import BaseImbalanceHandler
from arrange_library.models.library_info import EnhancedLibraryInfo

INPUT = "paiji_complete/tmp/59_sta_arrange_recheck57_rerun.csv"
handler = BaseImbalanceHandler()

def is_scheduled(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() != 'nan' and s != '0'

def pick(row: dict, *keys, default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default

def build_lib(row: dict) -> EnhancedLibraryInfo:
    payload = {
        'ORIGREC': pick(row, 'wkorigrec', 'origrec', default=''),
        'SAMPLEID': pick(row, 'wksampleid', 'sampleid', default=''),
        'SAMPLETYPECODE': pick(row, 'wksampletype', 'sampletypecode', default=''),
        'DATATYPE': pick(row, 'wkdatatype', 'datatype', default=''),
        'CUSTOMERLIBRARY': '是' if str(pick(row, 'wksampleid', default='')).startswith('FKDL') or str(pick(row, 'wksampletype', default='')).startswith('客户') else '否',
        'BASETYPE': '双',
        'NUMBEROFBASES': 10,
        'INDEXNUMBER': 1,
        'INDEXSEQ': pick(row, 'wkindexseq', 'indexseq', default=''),
        'ADDTESTSREMARK': pick(row, 'wkaddtestsremark', default=''),
        'PRODUCTLINE': pick(row, 'wkproductline', default='S'),
        'PEAKSIZE': pick(row, 'wkpeaksize', default=0),
        'EQTYPE': pick(row, 'wkeqtype', default='Nova X-25B'),
        'CONTRACTDATA_RAW': pick(row, 'wkcontractdata', default=0),
        'TESTCODE': 1001,
        'TESTNO': pick(row, 'wktestno', default='PE150'),
        'SUBPROJECTNAME': pick(row, 'wksubprojectname', default=''),
        'CREATE_DATE': pick(row, 'wkcreatedate', default=''),
        'DELIVERY_DATE': pick(row, 'wkdeliverydate', default=''),
        'LABTYPE': pick(row, 'wkdatatype', default=''),
        'DATAVOLUMETYPE': '标准',
        'BOARDNUMBER': pick(row, 'wkboardnumber', default=''),
    }
    lib = EnhancedLibraryInfo.create_from_dict(payload)
    lib.jjbj = '是' if str(pick(row, 'wk_jjbj', default='否')).strip() == '是' else '否'
    lib.sample_type_code = str(pick(row, 'wksampletype', 'wkdatatype', default='') or '')
    lib.data_type = str(pick(row, 'wkdatatype', 'wksampletype', default='') or '')
    return lib

df = pd.read_csv(INPUT)
if 'llaneid' not in df.columns:
    raise SystemExit('missing llaneid')
mask = df['llaneid'].map(is_scheduled)
df = df.loc[mask].copy()
reports = []
reason_counter = Counter()
for lane_id, sub in df.groupby('llaneid', sort=False):
    libs = [build_lib(row) for row in sub.to_dict(orient='records')]
    ok, reason = handler.check_mix_compatibility(libs, enforce_total_limit=False)
    reports.append({
        'lane_id': lane_id,
        'ok': bool(ok),
        'reason': reason,
        'types': sorted({str(getattr(lib, 'sample_type_code', '') or '') for lib in libs if str(getattr(lib, 'jjbj', '') or '') == '是'}),
        'imbalance_types': len({str(getattr(lib, 'sample_type_code', '') or '') for lib in libs if str(getattr(lib, 'jjbj', '') or '') == '是'}),
        'library_count': len(libs),
    })
    if not ok:
        reason_counter[reason] += 1

reports_sorted = sorted(reports, key=lambda x: (x['ok'], -x['imbalance_types'], x['lane_id']))
summary = {
    'lane_count': len(reports),
    'pass_count': sum(1 for r in reports if r['ok']),
    'fail_count': sum(1 for r in reports if not r['ok']),
    'reason_counts': dict(reason_counter),
    'top_failed': [
        {k: r[k] for k in ('lane_id','reason','imbalance_types','library_count')}
        for r in reports_sorted if not r['ok']
    ][:10],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
