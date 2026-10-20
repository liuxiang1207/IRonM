// 急症词表离线版（spec §21–22）。不依赖任何第三方库，断网时照常拦截。
// 数据由服务端同一份词表生成（emergency_terms.json，运行 make offline 更新）；
// 逻辑与 ironm/safety/emergency.py 逐步对应，两边结果必须完全一致（tests/safety/test_offline.py）。

export type Level = "emergency" | "red_flag" | "none";

export interface BloodPressure {
  context: string[];
  systolic_at_least: number;
  diastolic_at_least: number;
}

export interface Symptom {
  id: string;
  name: string;
  terms: string[];
  combos: string[][][];
  blood_pressure: BloodPressure | null;
}

export interface EmergencyRule {
  id: string;
  level: "emergency" | "red_flag";
  category: string;
  title: string;
  all: string[];
}

export interface Negation {
  before: string[];
  exceptions: string[];
  past_after: string;
  past_not_followed_by: string[];
  recurrence: string[];
}

export interface TermBook {
  format: number;
  separators: string;
  whitespace: string;
  version: string;
  review: "pending_review" | "approved";
  negation: Negation;
  symptoms: Symptom[];
  rules: EmergencyRule[];
}

export interface Assessment {
  level: Level;
  rule_ids: string[];
  categories: string[];
}

const RANK: Record<Level, number> = { none: 0, red_flag: 1, emergency: 2 };

// 统一文字（全角转半角、英文转小写）后按分句符号切开，再去掉空白；空分句丢弃。
export function clauses(book: TermBook, text: string): string[] {
  const unified = text.normalize("NFKC").toLowerCase();
  const parts: string[] = [];
  let current = "";
  for (const ch of unified) {
    if (book.separators.includes(ch)) {
      parts.push(current);
      current = "";
    } else if (!book.whitespace.includes(ch)) {
      current += ch;
    }
  }
  parts.push(current);
  return parts.filter((p) => p.length > 0);
}

function bpValue(digits: string): number | null {
  return digits.length >= 2 && digits.length <= 3 ? Number(digits) : null;
}

function bloodPressureMatches(bp: BloodPressure, clause: string): boolean {
  if (!bp.context.some((c) => clause.includes(c))) return false;
  const systolic: (number | null)[] = [];
  const diastolic: (number | null)[] = [];
  for (const m of clause.matchAll(/([0-9]+)\/([0-9]+)/g)) {
    systolic.push(bpValue(m[1]));
    diastolic.push(bpValue(m[2]));
  }
  for (const m of clause.matchAll(/高压([0-9]+)/g)) systolic.push(bpValue(m[1]));
  for (const m of clause.matchAll(/低压([0-9]+)/g)) diastolic.push(bpValue(m[1]));
  return (
    systolic.some((v) => v !== null && v >= bp.systolic_at_least) ||
    diastolic.some((v) => v !== null && v >= bp.diastolic_at_least)
  );
}

// clause 中 [start, end) 这一处说法算不算出现
function counts(neg: Negation, clause: string, start: number, end: number, pastCounts: boolean): boolean {
  const head = clause.slice(0, start);
  const negated = neg.before.some((w) => head.endsWith(w));
  if (negated && !neg.exceptions.some((w) => head.endsWith(w))) return false;
  const tail = clause.slice(end);
  if (pastCounts || !tail.startsWith(neg.past_after)) return true;
  const rest = tail.slice(neg.past_after.length);
  return neg.past_not_followed_by.some((w) => rest.startsWith(w));
}

function found(neg: Negation, clause: string, term: string, pastCounts: boolean): boolean {
  let i = clause.indexOf(term);
  while (i !== -1) {
    if (counts(neg, clause, i, i + term.length, pastCounts)) return true;
    i = clause.indexOf(term, i + 1);
  }
  return false;
}

function symptomMatches(symptom: Symptom, clause: string, neg: Negation, pastCounts: boolean): boolean {
  const f = (t: string) => found(neg, clause, t, pastCounts);
  if (symptom.terms.some(f)) return true;
  if (symptom.combos.some((combo) => combo.every((group) => group.some(f)))) return true;
  // 血压按数值判断，不受否定影响
  return symptom.blood_pressure !== null && bloodPressureMatches(symptom.blood_pressure, clause);
}

export function assess(book: TermBook, text: string): Assessment {
  const parts = clauses(book, text);
  const neg = book.negation;
  const pastCounts = parts.some((c) => neg.recurrence.some((w) => c.includes(w)));
  const present = new Set(
    book.symptoms.filter((s) => parts.some((c) => symptomMatches(s, c, neg, pastCounts))).map((s) => s.id),
  );
  const fired = book.rules.filter((r) => r.all.every((s) => present.has(s)));
  let level: Level = "none";
  for (const r of fired) if (RANK[r.level] > RANK[level]) level = r.level;
  return {
    level,
    rule_ids: fired.map((r) => r.id),
    categories: [...new Set(fired.map((r) => r.category))],
  };
}
