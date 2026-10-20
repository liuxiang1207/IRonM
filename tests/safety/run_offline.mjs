// 测试用：从标准输入读 {"book": 词表数据路径, "texts": [...]}，用离线脚本逐句判定，输出 JSON 数组。
import { readFileSync } from "node:fs";
import { assess } from "../../app/src/safety/emergency.ts";

const input = JSON.parse(readFileSync(0, "utf-8"));
const book = JSON.parse(readFileSync(input.book, "utf-8"));
process.stdout.write(JSON.stringify(input.texts.map((t) => assess(book, t))));
