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
  not_followed_by: string[];
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
  past_after: string | null;
  past_not_followed_by: string[];
  recurrence: string[];
  recent: string[];
}

export interface LevelText {
  conclusion: string;
  stop: string;
  cannot_judge: string;
  help: string;
}

export interface CategoryText {
  stop: string;
  department: string;
  bring: string[];
  conclusion: string | null;
  help_extra: string | null;
}

export interface Replies {
  version: string;
  review: "pending_review" | "approved";
  titles: LevelText;
  levels: Record<"emergency" | "red_flag", LevelText>;
  categories: Record<string, CategoryText>;
}

export interface ReplySection {
  kind: "conclusion" | "stop" | "cannot_judge" | "help";
  title: string;
  text: string;
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
  replies: Replies;
}

export interface Assessment {
  level: Level;
  rule_ids: string[];
  categories: string[];
}

const RANK: Record<Level, number> = { none: 0, red_flag: 1, emergency: 2 };

// 统一文字（全角转半角、英文转小写）后按分句符号切开，再去掉空白；空分句丢弃。
// 全角 ASCII（U+FF01–FF5E）转半角、全角空格转空格、小号与半角标点转常用写法、英文字母转小写，其余字符不变。
// 不用 normalize("NFKC")：各运行环境的 Unicode 版本不同，结果会与服务端不一致。
// 小号与半角标点转成对应的常用写法：｡ ﹐ ﹒ ﹔ ﹖ ﹗（与服务端 PUNCTUATION 相同）
const PUNCTUATION: Record<number, number> = {
  0xff61: 0x3002,
  0xfe50: 0x2c,
  0xfe52: 0x2e,
  0xfe54: 0x3b,
  0xfe56: 0x3f,
  0xfe57: 0x21,
};

export function unify(text: string): string {
  let out = "";
  for (const ch of text) {
    let code = ch.codePointAt(0) ?? 0;
    if (code >= 0xff01 && code <= 0xff5e) code -= 0xfee0;
    else if (code === 0x3000) code = 0x20;
    else code = PUNCTUATION[code] ?? code;
    if (code >= 0x41 && code <= 0x5a) code += 0x20;
    out += String.fromCodePoint(code);
  }
  return out;
}

export function clauses(book: TermBook, text: string): string[] {
  const unified = unify(text);
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
  if (pastCounts || neg.past_after === null || !tail.startsWith(neg.past_after)) return true;
  const rest = tail.slice(neg.past_after.length);
  return neg.past_not_followed_by.some((w) => rest.startsWith(w));
}

// 后面紧跟 notFollowedBy 中的字时这一处不算
function found(
  neg: Negation,
  clause: string,
  term: string,
  pastCounts: boolean,
  notFollowedBy: string[],
): boolean {
  let i = clause.indexOf(term);
  while (i !== -1) {
    const end = i + term.length;
    const followed = notFollowedBy.some((w) => clause.startsWith(w, end));
    if (!followed && counts(neg, clause, i, end, pastCounts)) return true;
    i = clause.indexOf(term, i + 1);
  }
  return false;
}

function symptomMatches(symptom: Symptom, clause: string, neg: Negation, pastCounts: boolean): boolean {
  const f = (t: string) => found(neg, clause, t, pastCounts, symptom.not_followed_by);
  if (symptom.terms.some(f)) return true;
  if (symptom.combos.some((combo) => combo.every((group) => group.some(f)))) return true;
  // 血压按数值判断，不受否定影响
  return symptom.blood_pressure !== null && bloodPressureMatches(symptom.blood_pressure, clause);
}

export function assess(book: TermBook, text: string): Assessment {
  const parts = clauses(book, text);
  const neg = book.negation;
  // 再次发生的字样对整句起作用，近期字样只对它所在的分句起作用
  const again = parts.some((c) => neg.recurrence.some((w) => c.includes(w)));
  const marked = parts.map((c): [string, boolean] => [c, again || neg.recent.some((w) => c.includes(w))]);
  const present = new Set(
    book.symptoms.filter((s) => marked.some(([c, r]) => symptomMatches(s, c, neg, r))).map((s) => s.id),
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

function unique(items: string[]): string[] {
  return [...new Set(items)];
}

// 红旗四段式：安全结论 → 现在需要停止的 → 我不能判断的 → 建议的专业帮助（与 ironm/safety/replies.py 对应）
export function fourSection(book: TermBook, assessment: Assessment): ReplySection[] {
  if (assessment.level === "none") return [];
  const r = book.replies;
  const level = r.levels[assessment.level];
  const cats = assessment.categories.map((c) => r.categories[c]);
  const stop = unique([level.stop, ...cats.map((c) => c.stop)]).join("");
  const departments = unique(cats.map((c) => c.department)).join("或");
  const bring = unique(cats.flatMap((c) => c.bring)).join("、");
  const extra = unique(cats.flatMap((c) => (c.help_extra ? [c.help_extra] : []))).join("");
  const help = extra + level.help.replace("{departments}", departments).replace("{bring}", bring);
  // 第一个带自己结论句的类别优先（按命中条目的顺序）
  const conclusion = cats.find((c) => c.conclusion)?.conclusion ?? level.conclusion;
  return [
    { kind: "conclusion", title: r.titles.conclusion, text: conclusion },
    { kind: "stop", title: r.titles.stop, text: stop },
    { kind: "cannot_judge", title: r.titles.cannot_judge, text: level.cannot_judge },
    { kind: "help", title: r.titles.help, text: help },
  ];
}
