// 测试用：从标准输入读 {"book": 离线数据路径, "texts": [...]}，用离线脚本逐句判定并生成四段式，输出 JSON 数组。
import { readFileSync } from "node:fs";
import { assess, fourSection } from "../../app/src/safety/emergency.ts";

const input = JSON.parse(readFileSync(0, "utf-8"));
const book = JSON.parse(readFileSync(input.book, "utf-8"));
const out = input.texts.map((t) => {
  const assessment = assess(book, t);
  return { assessment, sections: fourSection(book, assessment) };
});
process.stdout.write(JSON.stringify(out));
