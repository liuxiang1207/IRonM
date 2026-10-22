// IronM 网页版（spec [2026-10-21] §19）：只调用服务端已有的接口，规则与计算都在服务端。
// 浏览器里只保存登录凭证与用户编号，不保存健康数据（对话记录只在当前页面里）。
"use strict";

const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Shanghai";
const KEYS = { token: "ironm.token", user: "ironm.user", dev: "ironm.devUser" };
const view = () => document.getElementById("view");
const chatLog = []; // 本页对话：刷新即清空

// ---- 小工具 ----

function load(key) {
  try { return localStorage.getItem(KEYS[key]); } catch { return null; }
}
function save(key, value) {
  try { value === null ? localStorage.removeItem(KEYS[key]) : localStorage.setItem(KEYS[key], value); } catch { /* 浏览器不让存时只在本页有效 */ }
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "class") el.className = v;
    else if (v !== null && v !== undefined && v !== false) el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const card = (...children) => h("div", { class: "card" }, ...children);
const show = (...nodes) => view().replaceChildren(...nodes);

function uuid4() {
  // crypto.randomUUID 只在 https 或本机可用；局域网用手机访问时用 getRandomValues 生成
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
  const x = [...b].map((v) => v.toString(16).padStart(2, "0")).join("");
  return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`;
}

function detail(data) {
  if (!data) return "";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("；");
  return "";
}

async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  const token = load("token"), dev = load("dev");
  if (token) headers.Authorization = `Bearer ${token}`;
  else if (dev) headers["X-IronM-User"] = dev;
  const resp = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  let data = null;
  if (resp.status !== 204) { try { data = await resp.json(); } catch { data = null; } }
  if (resp.status === 401 && !path.startsWith("/auth/login")) {
    signOut(detail(data) || "请先登录");
    throw new Error("unauthorized");
  }
  if (!resp.ok) throw new Error(detail(data) || `请求没有成功（${resp.status}）`);
  return data;
}

function failed(err) {
  if (err.message === "unauthorized") return null;
  return h("p", { class: "error" }, err.message);
}

function event(type, payload) {
  const now = new Date().toISOString();
  return { event_id: uuid4(), user_id: load("user"), type, occurred_at: now, recorded_at: now,
    source: "user", confirmed: true, payload };
}

// ---- 登录 ----

function signOut(message) {
  save("token", null); save("user", null); save("dev", null);
  renderLogin(message);
}

async function renderLogin(message) {
  document.getElementById("bar").hidden = true;
  let options = { dev_user_login: false };
  try { options = await api("/auth/options"); } catch { /* 拿不到时只显示设备码登录 */ }
  const code = h("input", { name: "code", autocomplete: "one-time-code", required: true });
  const device = h("input", { name: "device", value: "网页浏览器", required: true });
  const err = h("p", { class: "error" }, message || "");
  const form = h("form", { class: "card", on: { submit: async (e) => {
    e.preventDefault();
    try {
      const got = await api("/auth/login", { method: "POST", body: { code: code.value.trim(), device: device.value.trim() } });
      save("token", got.token); save("user", got.user_id); save("dev", null);
      location.hash = "#today"; route();
    } catch (ex) { err.textContent = ex.message; }
  } } },
    h("h1", {}, "登录 IronM"),
    h("p", { class: "muted" }, "设备码由管理员生成，只能用一次。"),
    h("label", {}, "设备码", code), h("label", {}, "这台设备的名字", device),
    h("button", { class: "primary", type: "submit" }, "登录"), err);
  const nodes = [form];
  if (options.dev_user_login) {
    const who = h("input", { name: "user", placeholder: "例如 me" });
    nodes.push(h("form", { class: "card", on: { submit: (e) => {
      e.preventDefault();
      if (!who.value.trim()) return;
      save("dev", who.value.trim()); save("user", who.value.trim()); save("token", null);
      location.hash = "#today"; route();
    } } },
      h("h2", {}, "开发环境：直接用用户名进入"),
      h("p", { class: "muted" }, "只在开发与测试环境提供，正式环境只能用设备码登录。"),
      h("label", {}, "用户名", who), h("button", { type: "submit" }, "进入")));
  }
  show(...nodes);
}

// ---- 路由 ----

const VIEWS = { today: renderToday, chat: renderChat, progress: renderProgress, plan: renderPlan, me: renderMe };

async function route() {
  if (!load("token") && !load("dev")) return renderLogin();
  let plan;
  try { plan = await api("/plan"); } catch (ex) { return show(failed(ex)); }
  document.getElementById("bar").hidden = false;
  if (plan.status === "none") { markNav(null); return renderOnboarding(); }
  const name = (location.hash || "#today").slice(1);
  markNav(VIEWS[name] ? name : "today");
  (VIEWS[name] || renderToday)(plan.plan);
}

function markNav(name) {
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("on", a.dataset.view === name);
}

// ---- 建档 ----

const GOALS = { fitness: "健身", strength: "增肌 / 力量", fat_loss: "减脂", habits: "养成运动习惯", maintenance: "维持" };
const ENVS = { gym: "健身房", home: "家里", hotel: "酒店 / 出差" };
const LEVELS = { beginner: "新手", intermediate: "练过一段时间", advanced: "经验丰富" };
const ACTIVITY = { sedentary: "久坐或轻度", moderate: "中度", vigorous: "重度" };

function select(name, choices, value) {
  return h("select", { name }, ...Object.entries(choices).map(([v, t]) => h("option", { value: v, selected: v === value }, t)));
}

// 出生日期：年、月、日三个下拉框（浏览器自带的日期控件选年份要往前翻很久）。
// 取值为 YYYY-MM-DD；没选全或日期不存在（如 2 月 30 日）时为空字符串
function birthPicker(name) {
  const thisYear = new Date().getFullYear();
  const pick = (part, label, values, unit) => h("select", { name: `${name}_${part}`, required: true, "aria-label": label },
    h("option", { value: "" }, label), ...values.map((v) => h("option", { value: v }, `${v} ${unit}`)));
  const year = pick("year", "年", Array.from({ length: 100 }, (_, i) => thisYear - i), "年");
  const month = pick("month", "月", Array.from({ length: 12 }, (_, i) => i + 1), "月");
  const day = pick("day", "日", Array.from({ length: 31 }, (_, i) => i + 1), "日");
  const box = h("span", { class: "row", "data-picker": name }, year, month, day);
  Object.defineProperty(box, "value", { get() {
    if (!year.value || !month.value || !day.value) return "";
    const d = new Date(Date.UTC(Number(year.value), Number(month.value) - 1, Number(day.value)));
    if (d.getUTCMonth() !== Number(month.value) - 1) return "";
    return d.toISOString().slice(0, 10);
  } });
  return box;
}

function renderOnboarding(result) {
  const f = (name, attrs = {}) => h("input", { name, ...attrs });
  const fields = {
    birth_date: birthPicker("birth_date"),
    sex: select("sex", { male: "男", female: "女" }, "male"),
    height_cm: f("height_cm", { type: "number", step: "0.1", min: 100, max: 250, required: true }),
    weight_kg: f("weight_kg", { type: "number", step: "0.1", min: 25, max: 300, required: true }),
    waist_cm: f("waist_cm", { type: "number", step: "0.1", min: 40, max: 200, placeholder: "可不填" }),
    goal: select("goal", GOALS, "fitness"),
    environment: select("environment", ENVS, "gym"),
    available_days: select("available_days", { 1: "1 天", 2: "2 天", 3: "3 天", 4: "4 天", 5: "5 天", 6: "6 天", 7: "7 天" }, "3"),
    training_level: select("training_level", LEVELS, "beginner"),
    activity: select("activity", ACTIVITY, "moderate"),
  };
  const questions = h("div", { class: "card", hidden: !result }, h("h2", {}, "还需要补充"));
  const answers = {};
  const out = h("div", {});
  const form = h("form", { class: "card", on: { submit: async (e) => {
    e.preventDefault();
    // 健康状况不替用户默认“都没有”：由服务端追问，用户必须自己勾选（含“都没有”）
    const body = { restrictions: [] };
    if (!fields.birth_date.value) {
      out.replaceChildren(h("p", { class: "error" }, "出生日期不存在，请重新选择（比如 2 月没有 30 日）。"));
      return;
    }
    for (const [k, el] of Object.entries(fields)) {
      if (el.value === "") continue;
      body[k] = el.type === "number" || k === "available_days" ? Number(el.value) : el.value;
    }
    const picked = collect(answers);
    if (Object.keys(picked).length) body.answers = picked;
    out.replaceChildren(h("p", { class: "muted" }, "正在生成第一版计划……"));
    try {
      const r = await api(`/onboarding?tz=${encodeURIComponent(TZ)}`, { method: "POST", body });
      if (r.status === "plan") { location.hash = "#today"; return route(); }
      if (r.status === "questions") { showQuestions(questions, answers, r.questions); out.replaceChildren(); return; }
      // 不能给计划：显示服务端给的原因（审定过的说明文字），不显示代号
      const why = r.notice_texts && r.notice_texts.length ? r.notice_texts : ["目前不能给你出训练计划。"];
      out.replaceChildren(h("div", { class: "warn" }, ...why.map((t) => h("div", {}, t))),
        h("p", { class: "muted" }, "如果资料填错了，改正后重新提交即可。"));
    } catch (ex) { out.replaceChildren(failed(ex)); }
  } } },
    h("h1", {}, "建档"),
    h("p", { class: "muted" }, "填一下基本情况，就能得到第一版计划。健康状况会在下一步问，请如实勾选。"),
    h("div", { class: "grid" },
      h("label", {}, "出生日期", fields.birth_date), h("label", {}, "性别", fields.sex),
      h("label", {}, "身高（厘米）", fields.height_cm), h("label", {}, "体重（公斤）", fields.weight_kg),
      h("label", {}, "腰围（厘米）", fields.waist_cm), h("label", {}, "目标", fields.goal),
      h("label", {}, "在哪练", fields.environment), h("label", {}, "每周能练几天", fields.available_days),
      h("label", {}, "训练经验", fields.training_level), h("label", {}, "日常活动量", fields.activity)),
    questions,
    h("button", { class: "primary", type: "submit" }, "生成计划"), out);
  show(form);
}

function showQuestions(box, answers, questions) {
  box.hidden = false;
  box.replaceChildren(h("h2", {}, "还需要补充"));
  for (const q of questions) {
    const id = q.question_id;
    const multi = /conditions$|restrictions$/.test(id);
    if (q.options && q.options.length) {
      const options = /conditions$/.test(id) ? [...q.options, { value: "none", label: "都没有" }] : q.options;
      const inputs = options.map((o) => h("label", { class: "inline" },
        h("input", { type: multi ? "checkbox" : "radio", name: id, value: o.value }), o.label));
      answers[id] = { multi, inputs };
      box.append(h("div", {}, h("p", {}, q.text, q.blocking ? "" : "（可以跳过）"), h("div", { class: "row" }, ...inputs)));
    } else {
      const input = /date$/.test(id) ? birthPicker(id)
        : h("input", { name: id, type: /_(cm|kg)$/.test(id) ? "number" : "text", step: "0.1" });
      answers[id] = { input };
      box.append(h("label", {}, q.text + (q.blocking ? "" : "（可以跳过）"), input));
    }
  }
}

function collect(answers) {
  const out = {};
  for (const [id, a] of Object.entries(answers)) {
    if (a.inputs) {
      const on = a.inputs.map((l) => l.querySelector("input")).filter((i) => i.checked).map((i) => i.value);
      if (a.multi) { if (on.length) out[id] = on.filter((v) => v !== "none"); }
      else if (on.length) out[id] = on[0];
    } else if (a.input.value !== "") {
      out[id] = a.input.type === "number" ? Number(a.input.value) : a.input.value;
    }
  }
  return out;
}

// ---- 今日 ----

const STATUS = { done: "完成", half: "练一半", skipped: "没练" };
const LEVEL_TEXT = { 1: "很差", 2: "疲惫", 3: "一般", 4: "良好", 5: "充沛" };

async function renderToday() {
  show(h("p", { class: "muted" }, "正在读取今日任务……"));
  let today, energy;
  try {
    [today, energy] = await Promise.all([
      api(`/plan/today?tz=${encodeURIComponent(TZ)}`), api(`/energy/today?tz=${encodeURIComponent(TZ)}`)]);
  } catch (ex) { return show(failed(ex)); }
  const nodes = [h("h1", {}, `今日 · ${today.date}`)];
  if (today.greeting) nodes.push(h("div", { class: "note" }, today.greeting, today.next_step ? ` ${today.next_step}` : ""));
  else if (today.next_step) nodes.push(h("div", { class: "note" }, today.next_step));
  if (today.mode) nodes.push(h("div", { class: "note" }, `${today.mode.mode === "travel" ? "出差" : "节假日"}模式，到 ${today.mode.until} 结束`));
  if (today.diet_note) nodes.push(h("div", { class: "note" }, today.diet_note));

  const training = card(h("h2", {}, today.training.scheduled ? "今天要练" : "今天不安排训练"));
  for (const it of today.training.items) {
    training.append(h("div", {},
      h("b", {}, it.exercise), ` ${it.sets} 组 × ${it.reps[0]}–${it.reps[1]} 次，用力程度 RPE ${it.rpe[0]}–${it.rpe[1]}`,
      it.key_points.length ? h("ul", {}, ...it.key_points.map((k) => h("li", {}, k))) : null,
      it.cautions && it.cautions.length ? h("div", { class: "muted" }, "注意：", it.cautions.join("；")) : null));
  }
  if (today.minimum_task && today.minimum_task.length) {
    training.append(h("details", {}, h("summary", {}, "最低任务（没时间或状态差时做这些就好）"),
      h("ul", {}, ...today.minimum_task.map((m) => h("li", {}, `${m.exercise} ${m.sets} 组`)))));
  }
  training.append(h("p", { class: "muted" }, `本周有氧目标 ${today.weekly_cardio_minutes ?? "—"} 分钟`
    + (today.steps_target ? `；每天 ${today.steps_target} 步` : "")));
  const msg = h("span", { class: "muted" });
  if (today.training.scheduled) {
    training.append(h("div", { class: "row" }, "打卡：",
      ...Object.entries(STATUS).map(([v, t]) => h("button", { on: { click: () => send(event("workout_check", { status: v }), msg, `已记下：${t}`) } }, t)), msg));
  }
  nodes.push(training);
  nodes.push(energyCard(energy));
  nodes.push(weightCard(today.weigh_in_today));
  if (today.checkin_today) nodes.push(checkinCard());
  show(...nodes);
}

async function send(ev, msg, ok) {
  try { await api("/events", { method: "POST", body: ev }); msg.textContent = ok; }
  catch (ex) { msg.textContent = ex.message; }
}

function energyCard(energy) {
  const box = card(h("h2", {}, "今天的精力"));
  const msg = h("span", { class: "muted" });
  box.append(h("div", { class: "row" }, ...Object.entries(LEVEL_TEXT).map(([v, t]) =>
    h("button", { class: energy.selected && String(energy.level) === v ? "on" : "", on: { click: async () => {
      await send(event("energy_check", { level: Number(v) }), msg, "已记下");
      renderToday();
    } } }, `${v} ${t}`)), msg));
  if (energy.selected && energy.adjustment) {
    const a = energy.adjustment;
    const parts = [`训练量按 ${Math.round(a.volume_factor * 100)}%`];
    if (a.rpe_cap_max) parts.push(`用力程度不超过 RPE ${a.rpe_cap_max}`);
    if (a.max_minutes) parts.push(`不超过 ${a.max_minutes} 分钟`);
    box.append(h("div", { class: "note" }, `今天${parts.join("，")}。`), a.safety_hint ? h("div", { class: "warn" }, a.safety_hint) : null);
  }
  return box;
}

function weightCard(isDay) {
  const kg = h("input", { type: "number", step: "0.1", min: 20, max: 400, placeholder: "公斤" });
  const msg = h("span", { class: "muted" });
  return card(h("h2", {}, isDay ? "今天是称重日" : "记体重"),
    h("div", { class: "row" }, kg, h("button", { on: { click: () => {
      if (kg.value) send(event("weight", { weight_kg: Number(kg.value) }), msg, "已记下");
    } } }, "记下"), msg));
}

function checkinCard() {
  const score = (name) => select(name, { "": "选一下", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5" }, "");
  const fatigue = score("fatigue"), sleep = score("sleep_quality");
  const full = select("record_completeness", { full: "这周都记了", partial: "只记录了一部分" }, "full");
  const msg = h("span", { class: "muted" });
  return card(h("h2", {}, "今天是签到日"),
    h("p", { class: "muted" }, "1 分最低、5 分最高。签到后会做这周的复盘。"),
    h("div", { class: "grid" }, h("label", {}, "这周的疲劳", fatigue), h("label", {}, "这周的睡眠", sleep), h("label", {}, "记录", full)),
    h("div", { class: "row" }, h("button", { class: "primary", on: { click: () => {
      const payload = { record_completeness: full.value };
      if (fatigue.value) payload.fatigue = Number(fatigue.value);
      if (sleep.value) payload.sleep_quality = Number(sleep.value);
      send(event("checkin", payload), msg, "已签到");
    } } }, "签到"), msg));
}

// ---- 对话 ----

const PART = { conclusion: "结论", for_you: "针对你", why: "为什么", how: "怎么做", caution: "注意",
  uncertain: "不确定", answer: "", notice: "", clarify: "" };

function renderChat() {
  const list = h("div", { class: "msg" });
  const input = h("textarea", { placeholder: "想问训练、饮食、恢复，或说说今天的情况", maxlength: 2000 });
  const form = h("form", { class: "card", on: { submit: async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    chatLog.push({ me: text });
    draw(list);
    try { chatLog.push({ reply: await api("/chat", { method: "POST", body: { text } }) }); }
    catch (ex) { chatLog.push({ error: ex.message }); }
    draw(list);
  } } }, input, h("button", { class: "primary", type: "submit" }, "发送"));
  draw(list);
  show(h("h1", {}, "对话"), list, form);
}

function draw(list) {
  list.replaceChildren(...chatLog.map((m) => {
    if (m.me) return h("div", { class: "msg me" }, m.me);
    if (m.error) return h("p", { class: "error" }, m.error);
    return reply(m.reply);
  }));
}

function reply(r) {
  const box = card();
  if (r.sections && r.sections.length) {
    box.append(h("div", { class: "warn" }, ...r.sections.map((s) => h("div", {}, h("b", {}, s.title), " ", s.text))));
  }
  for (const p of r.answer || []) {
    box.append(h("div", { class: "part" }, PART[p.kind] ? h("b", {}, PART[p.kind]) : null, p.text));
  }
  if (r.citations && r.citations.length) {
    box.append(h("details", {}, h("summary", {}, `依据（${r.citations.length} 条）`),
      h("ul", {}, ...r.citations.map((c) => h("li", {}, c.label, c.detail ? h("div", { class: "muted" }, c.detail) : null)))));
  }
  for (const c of r.candidates || []) box.append(candidate(c));
  return box;
}

function candidate(c) {
  const msg = h("span", { class: "muted" });
  const decide = (d) => api(`/profile/candidates/${encodeURIComponent(c.candidate_id)}/${d}`, { method: "POST" })
    .then(() => { msg.textContent = d === "confirm" ? "已更新资料" : "已忽略"; }, (ex) => { msg.textContent = ex.message; });
  return h("div", { class: "note row" }, `要把资料里的「${c.field}」改成「${c.value}」吗？`,
    h("button", { on: { click: () => decide("confirm") } }, "确认"),
    h("button", { on: { click: () => decide("reject") } }, "不对"), msg);
}

// ---- 进展 ----

const METRIC = { weight_trend: "体重趋势", diet_kcal_mean_7d: "近 7 天日均热量" };
const UNIT = { "kg/week": "千克 / 周", kcal: "千卡", "kcal/day": "千卡 / 天" };
const NEEDED = { days: "天的称重记录", weight_record: "次称重", span_days: "天的记录跨度",
  diet_days: "天的饮食记录", weekend_diet_days: "个周末日的饮食记录" };
const num = (v, unit) => (unit === "kg/week" ? v.toFixed(2) : String(Math.round(v)));

async function renderProgress() {
  let m;
  try { m = await api(`/metrics?tz=${encodeURIComponent(TZ)}`); } catch (ex) { return show(failed(ex)); }
  const nodes = [h("h1", {}, "进展")];
  for (const x of m.metrics) {
    const known = typeof x.value === "number";
    const ok = x.status === "ok";
    const unit = UNIT[x.unit] || x.unit;
    const value = known ? `${ok ? "" : "目前估算 "}${num(x.value, x.unit)} ${unit}` : "数据还不够";
    const box = card(h("h2", {}, METRIC[x.name] || x.name),
      h("div", {}, value, h("span", { class: "muted" }, `　精度 ${x.precision}`)));
    if (!ok && known) box.append(h("div", { class: "muted" }, "记录还不够，这个数还不能当结论。"));
    if (known && x.ci) {
      box.append(h("div", { class: "muted" }, `可信范围 ${num(x.ci.low, x.unit)} 到 ${num(x.ci.high, x.unit)} ${unit}`));
    }
    if (x.needed && x.needed.length) {
      box.append(h("div", { class: "note" }, "还差：", x.needed.map((n) => `${n.count} ${NEEDED[n.item] || n.item}`).join("，")));
    }
    nodes.push(box);
  }
  show(...nodes);
}

// ---- 计划 ----

const WEEKDAY = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
const FIELD = { weekly_cardio_minutes: "每周有氧（分钟）", session_slots: "每个动作的组数与次数",
  daily_kcal_target: "每日热量（千卡）", training_days_per_week: "每周训练天数", training_schedule: "训练日",
  intensity_target: "用力程度", diet_mode: "饮食记录方式" };
const PATTERN = { squat: "下肢蹲", hinge: "髋铰链", lunge: "单腿", push: "上肢推", pull: "上肢拉", core: "核心", cardio: "心肺" };

function fieldText(name, value) {
  if (value === null || value === undefined) return "—";
  if (name === "training_schedule" || name === "weigh_in_days") return value.map((d) => WEEKDAY[d]).join("、");
  if (name === "session_slots") return value.map((s) => `${PATTERN[s.pattern] || s.pattern} ${s.sets}×${s.reps[0]}–${s.reps[1]}`).join("；");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

async function renderPlan(plan) {
  let history = [], decisions = [];
  try { [history, decisions] = await Promise.all([api("/plan/history"), api("/plan/decisions")]); }
  catch (ex) { return show(failed(ex)); }
  const f = plan.fields;
  const current = card(h("h2", {}, `当前计划 · 第 ${plan.version} 版`),
    h("div", { class: "scroll" }, h("table", {}, ...Object.keys(FIELD).filter((k) => k in f).map((k) =>
      h("tr", {}, h("th", {}, FIELD[k]), h("td", {}, fieldText(k, f[k])))))));
  const cards = card(h("h2", {}, "依据卡"), ...plan.cards.map((c) => h("details", {},
    h("summary", {}, `${FIELD[c.field] || c.field}：${fieldText(c.field, c.to_value)}`),
    h("ul", {}, ...(c.facts || []).map((x) => h("li", {}, x))),
    c.source ? h("div", { class: "muted" }, `依据：${c.source.title}（${c.source.ref}）`) : null)));
  const reviews = card(h("h2", {}, "复盘记录"), decisions.length ? null : h("p", { class: "muted" }, "还没有复盘。"),
    ...decisions.map((d) => h("div", {}, h("b", {}, d.period), " ", d.summary || "", d.suggestion ? h("div", { class: "note" }, d.suggestion) : null)));
  const versions = card(h("h2", {}, "历史版本"), ...history.slice().reverse().map((v) => h("details", {},
    h("summary", {}, `第 ${v.version} 版 · ${v.created_at.slice(0, 10)} · ${v.reason}`),
    h("ul", {}, ...(v.changes || []).map((c) => h("li", {}, `${FIELD[c.field] || c.field}：${fieldText(c.field, c.from)} → ${fieldText(c.field, c.to)}`))))));
  show(h("h1", {}, "计划"), current, reviews, cards, versions);
}

// ---- 我的 ----

async function renderMe() {
  const nodes = [h("h1", {}, "我的"), h("p", { class: "muted" }, `用户：${load("user") || "—"}`)];
  if (load("token")) {
    try {
      const devices = await api("/auth/devices");
      const box = card(h("h2", {}, "已登录的设备"));
      for (const d of devices) {
        box.append(h("div", { class: "row" }, `${d.name}（${d.created_at.slice(0, 10)}）`, h("button", { on: { click: async () => {
          await api(`/auth/devices/${encodeURIComponent(d.device_id)}`, { method: "DELETE" });
          renderMe();
        } } }, "撤销")));
      }
      nodes.push(box);
    } catch (ex) { nodes.push(failed(ex)); }
  }
  nodes.push(h("button", { on: { click: () => signOut("已退出登录") } }, "退出登录"));
  show(...nodes);
}

window.addEventListener("hashchange", route);
route();
