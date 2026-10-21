"""急症词表与匹配（spec [2026-10-20] §21）。红旗测试集本身见 test_red_flags.py。"""

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.safety.emergency import (
    TERMS_PATH,
    Assessment,
    TermLoadError,
    assess,
    clauses,
    load_terms,
)
from tests.test_api_events import NOW

# 不在红旗测试集里的改写说法：同义词、换语序、他人发生。防止词表只贴合测试集。
PARAPHRASES = [
    ("跑着跑着胸口剧痛，到现在都没缓解", "emergency"),
    ("心口疼得厉害，左手臂也麻", "emergency"),
    ("胸前闷得慌，一直冒冷汗", "emergency"),
    ("练完突然昏过去了", "emergency"),
    ("我妈打羽毛球时突然倒地，叫不醒", "emergency"),
    ("刚才看到有人在跑道上晕倒了", "emergency"),
    ("有个会员在跑步机上抽搐", "emergency"),
    ("高温天跑完十公里，人有点神志不清", "emergency"),
    ("嘴唇发青，喘不过气", "emergency"),
    ("深蹲时突然头疼得受不了，从没这么疼过", "emergency"),
    ("右边半边脸麻了，胳膊也没劲", "emergency"),
    ("昨天练了三百个深蹲，今天小便是浓茶色", "emergency"),
    ("我是一型糖尿病，刚跑完出了一身虚汗", "emergency"),
    ("孕30周，今天跳操后感觉胎动变少了", "emergency"),
    ("跑完步高压190，头很晕", "emergency"),
    ("一侧小腿肿起来了，按着疼", "emergency"),
    ("腰痛，大小便都控制不住了", "emergency"),
    ("最近练腿老是头晕", "red_flag"),
    ("这周跑步时心脏乱跳了好几次", "red_flag"),
    ("跑步时左胸口有点闷，停下来就好", "red_flag"),
    ("爬三楼就气短，比以前差很多", "red_flag"),
    ("打篮球崴了脚，肿得不能着地", "red_flag"),
    ("最近吃完饭会抠喉咙吐出来", "red_flag"),
    ("腰疼得厉害，左腿一直发麻", "red_flag"),
    ("膝盖肿了好几天，摸着发烫", "red_flag"),
    ("肩膀疼得睡不着", "red_flag"),
    ("量了血压 185/100，没什么感觉", "red_flag"),
    # 审定后补充（spec §26）
    ("最近真的不想活了", "red_flag"),
    ("我割腕了", "emergency"),
    ("练完全身起风团，喉咙发紧", "emergency"),
    ("运动后全身起风团，喘不上气", "emergency"),
    ("跑完身上起了荨麻疹", "red_flag"),
    ("哮喘发作了，喷了药也不管用", "emergency"),
    ("跑步时哮喘犯了", "red_flag"),
    ("头撞到杠铃，吐了好几次", "emergency"),
    ("卧推时杠铃砸到头了", "red_flag"),
    ("突然一只眼睛黑了看不清", "emergency"),
    ("右手突然没劲，拿不住东西", "emergency"),
    ("血压190/115，喘不上气", "emergency"),
    ("我在吃格列美脲，练完发抖出冷汗", "emergency"),
    ("怀孕了，羊水破了", "emergency"),
    ("怀孕5个月，跑步时有点头晕", "red_flag"),
    ("我中暑了，头痛恶心", "red_flag"),
    ("大太阳下跑步晕倒了", "emergency"),
    ("今天吃了泻药", "red_flag"),
    ("吐血了", "emergency"),
    ("最近大便是黑的，柏油样", "red_flag"),
    ("打了减肥针，肚子疼到后背", "red_flag"),
    # 冲刺 2 末独立审查发现的漏判（spec §32）
    ("胸口、左胳膊都疼，出冷汗", "emergency"),
    ("胸口、后背都很疼", "emergency"),
    ("左边脸、手都麻了", "emergency"),
    ("嘴唇、眼睛都肿了，喘不过气", "emergency"),
    ("喉咙肿过敏", "emergency"),
    ("吃了虾嘴唇肿过敏了，喘不上气", "emergency"),
    ("吐血过多", "emergency"),
    ("胸痛过后一直喘不上气", "emergency"),
    ("胸口疼过来的时候还出冷汗", "emergency"),
    ("今天晕倒过一次", "emergency"),
    ("我想去死", "red_flag"),
    ("我爸说话不太清楚，嘴有点歪", "emergency"),
    ("突然头疼得厉害，从来没这么疼过", "emergency"),
    ("他突然倒了，叫他也不应", "emergency"),
    # 冲刺 2 复核：收窄说法后仍须判出的真实情况
    ("他突然倒了，喊他没反应", "emergency"),
    ("我朋友跑着跑着栽倒了", "emergency"),
    ("他倒在地上，喊了半天叫不醒", "emergency"),
    ("跑完喘不上气，感觉要窒息了", "emergency"),
    ("吃了花生嘴唇有点肿", "red_flag"),
    ("我爸讲话不太清楚", "emergency"),
    ("左边脸有点歪", "emergency"),
    ("坐着也喘不上气", "red_flag"),
    ("最近走几步就喘不过气", "red_flag"),
    # 冲刺 2 第四轮复核：直白说法、插字说法、条目范围
    ("胸口有点闷，突然下起雨来", "red_flag"),  # 突发胸痛须在同一分句
    ("跑完出了一身汗，胸口有点闷", "red_flag"),  # 运动出汗不是冷汗
    ("突然想起来昨天胸口疼过，今天好了", "red_flag"),
    ("胸口有点闷，突然想起来今天还没吃药", "red_flag"),
    ("健身持续了三个月，最近胸口有点闷", "red_flag"),
    ("后背练完很酸，胸口也有点紧", "red_flag"),
    ("左肩推完有点疼，胸口也紧", "red_flag"),
    ("体重秤上的数字看得我头晕眼花", "red_flag"),
    ("昨天吃火锅辣得嘴唇肿了", "red_flag"),
    ("我爸刚才在跑步机上突然一头栽下来，叫他也不答应", "emergency"),
    ("我同事在健身房晕了，喊她没反应", "emergency"),
    ("朋友跑步时倒在地上抽了几下，现在叫不醒", "emergency"),
    ("旁边的大哥举着举着突然躺地上不动了", "emergency"),
    ("练完腿起身的时候眼前一黑就晕了", "emergency"),
    ("我刚才突然胸口像被石头压着，喘不上气", "emergency"),
    ("胸口闷闷的，左边胳膊有点麻", "emergency"),
    ("心口那里一阵一阵绞着疼，出了一身汗", "emergency"),
    ("我妈说话突然大舌头，右手拿不住筷子", "emergency"),
    ("我爸早上起来一边脸麻了，嘴角往下耷拉", "emergency"),
    ("我老婆刚刚说话说着说着就含含糊糊的", "emergency"),
    ("跑完步右腿突然没力气，抬不起来", "emergency"),
    ("吃了花生酱以后喉咙紧紧的，有点喘", "emergency"),
    ("被蜜蜂蛰了，现在全身起红疙瘩，呼吸有点困难", "emergency"),
    ("吃完海鲜嘴巴和舌头都肿了", "emergency"),
    ("现在坐着不动也觉得喘不上气", "red_flag"),
    ("晚上睡觉平躺就喘，得坐起来才舒服", "red_flag"),
    ("跑步时胸口疼过好多次了，今天想问能不能继续", "red_flag"),
    ("以前胸口疼过，这两天爬楼梯也有点闷", "red_flag"),
    ("我之前胸闷过，昨天跑步又胸闷了", "red_flag"),
    ("头疼得像要裂开一样，突然开始的", "emergency"),
    ("举重的时候头突然炸裂一样疼", "emergency"),
    ("大热天跑完步人迷迷糊糊的，不怎么出汗了", "emergency"),
    ("练完腿第二天尿是深褐色的", "emergency"),
    ("我有糖尿病，跑完步手抖出冷汗", "emergency"),
    ("我腰疼了好几天，今天上厕所发现尿不出来", "emergency"),
    ("左小腿又肿又疼，摸着烫", "emergency"),
    ("刚刚跑步时心脏突然乱跳，现在还没好", "red_flag"),
    ("我爸在家里摔倒了，叫他他不答应", "emergency"),
    ("我朋友喝了酒在路边倒下了，怎么叫都叫不醒", "emergency"),
    ("我奶奶刚刚中风了，嘴都歪了", "emergency"),
    ("我爸好像中风了", "emergency"),
    ("感觉自己心梗了，胸口特别疼", "emergency"),
    ("教练倒地后没有呼吸了", "emergency"),
    ("他没有呼吸也没有心跳了", "emergency"),
    ("跑步时突然左膝盖疼，现在走不了路", "red_flag"),
    ("跑步的时候胸口疼，休息一下就好了", "red_flag"),
    ("刚才跑着跑着胸口疼，现在坐下也不缓解", "emergency"),
    ("我爸去年做过支架，今天爬山时胸口疼", "red_flag"),
    ("跑完步一直咳嗽，嗓子里有哨音，喷了药也没用", "emergency"),
    ("今天有点想不开，想从楼上跳下去", "red_flag"),
    ("练完背疼得我想死", "red_flag"),
    ("卷腹做到肚子疼得厉害", "red_flag"),
    ("有氧做到快晕过去了", "red_flag"),
    # 冲刺 2 第三轮复核：漏判与“以前发生过”
    ("他晕倒半小时才醒", "emergency"),
    ("他失去意识半分钟", "emergency"),
    ("他突然倒地出血了", "emergency"),
    ("跑步时晕倒进了医院", "emergency"),
    ("他一下子倒了下去叫不醒", "emergency"),
    ("今天跑步之前胸口疼过一阵", "red_flag"),
    ("刚才跑步以前晕倒过一次", "emergency"),
    ("上次跑步胸口疼过，今天又疼了", "red_flag"),
    ("动不动就喘", "red_flag"),
    ("平躺就喘不上气", "red_flag"),
    ("坐在沙发上都喘不过气", "red_flag"),
    ("跑完步突然喘不上气，休息了也不行", "emergency"),
    ("跑步身上痒，还有点头晕", "red_flag"),
    ("吃了芒果眼睛肿了", "red_flag"),
    ("被蜜蜂蛰了，脸肿，喘不上气", "emergency"),
    ("我爸嘴歪一点了，口水往下流", "emergency"),
    ("嘴歪的厉害，说话也不清", "emergency"),
    ("突然右边脸麻，嘴角往左歪", "emergency"),
    # 冲刺 2 二次复核：收窄后漏判的真实情况与“以前发生过”的分句
    ("他跑着跑着就倒了，叫他也不答应", "emergency"),
    ("他突然倒了，叫他不答应", "emergency"),
    ("跑完步人就晕了过去", "emergency"),
    ("他躺在地上一动不动", "emergency"),
    ("人没意识了", "emergency"),
    ("他失去知觉了", "emergency"),
    ("眼前一黑就倒了", "emergency"),
    ("跑步机上有个人倒了，怎么叫都没反应", "emergency"),
    ("他一头栽倒在跑步机上", "emergency"),
    ("突然倒了下去", "emergency"),
    ("我爸突然脸歪了", "emergency"),
    ("他半边脸歪了", "emergency"),
    ("嘴巴歪了，口水往下流", "emergency"),
    ("嘴角往一边歪", "emergency"),
    ("嘴有点歪，说话也不清楚", "emergency"),
    ("我爸说话突然不清楚了", "emergency"),
    ("说话有点含糊，舌头不灵活", "emergency"),
    ("话都说不清楚了", "emergency"),
    ("他一边脸下垂了", "emergency"),
    ("左边嘴角歪了", "emergency"),
    ("我妈突然呼吸不上来了", "emergency"),
    ("喘不过来，憋得要死", "emergency"),
    ("喘不过气，感觉要窒息了", "emergency"),
    ("坐着都喘", "red_flag"),
    ("躺下就喘不上气，得坐起来", "red_flag"),
    ("休息的时候也喘不上气", "red_flag"),
    ("气都喘不上来，什么都没干", "red_flag"),
    ("晚上睡觉憋醒好几次", "red_flag"),
    ("吃了花生脸和眼睛都肿了，呼吸困难", "emergency"),
    ("吃完虾脸肿得厉害，喘不上气", "emergency"),
    ("眼睛肿得睁不开，身上痒", "red_flag"),
    ("被蜜蜂蛰了脸肿得厉害", "red_flag"),
    ("吃了海鲜嘴唇肿了，喉咙发紧", "emergency"),
    ("吃了海鲜嘴唇肿了", "red_flag"),
    ("嘴肿了，全身起红疙瘩", "red_flag"),
    ("以前胸口疼过，今天跑步时也疼", "red_flag"),
    ("胸口疼过好几次，每次跑步都疼", "red_flag"),
    ("跑步时胸口疼过好几次", "red_flag"),
    ("上周晕倒过，今天又晕倒了", "emergency"),
    ("今天晕倒过一次", "emergency"),
    ("刚才又胸口疼，一直疼", "emergency"),
    ("这周胸口闷过两次", "red_flag"),
    ("以前晕倒过，刚才跑步时也晕倒了", "emergency"),
    ("今天跑步的时候，晕倒过一次", "emergency"),
    ("我晕倒过一次，就在刚才", "emergency"),
    ("昨天在健身房，胸口疼过一阵，现在还闷", "emergency"),
    ("这周跑步的时候，胸口疼过两次", "red_flag"),
    ("刚才练腿，突然晕倒过去了", "emergency"),
    ("我喘不上气，快憋死了", "emergency"),
    ("我胸痛，没有缓解", "emergency"),
    ("跑完步胸口压着疼，休息了也不行", "emergency"),
]

# 含关键字但没有症状：健身语境里的常见说法不能被拦下
ORDINARY = [
    "深蹲动作变形了怎么办",
    "我要去吐槽一下教练",
    "一边跑步一边听歌，小腿有点酸",
    "练完胸部好紧",
    "今天练胸和背，后背有点酸",
    "卧推要不要憋气",
    "跑步时心跳很快正常吗",
    "血压计哪个牌子好",
    "晕车好难受",
    # 审定后修正的误报（spec §26）
    "发力时把气吐出来",
    "呼气时把气吐掉",
    "吃完就吐槽教练",
    "我有胰岛素抵抗，练完腿有点酸",
    "早上迷迷糊糊起来去跑步",
    "喝了可乐老想尿尿",
    "手机没有反应了",
    "我还在坚持跑步",
    # 冲刺 2 复核发现的误报（spec §33）
    "他不应该这么练",
    "人不应该节食太狠吧",
    "一个人不应该天天称体重",
    "我给她倒了杯水",
    "他倒了一杯蛋白粉",
    "他没反应过来我说的话",
    "他没回应我的消息",
    "她不答应我加练",
    "脸歪了吗这个自拍",
    "嘴歪笑",
    "教练说话我听不清楚",
    "他讲话含糊其辞",
    "健身房太闷了，快憋死了",
    "这个动作让人有窒息感",
    "早上起来脸有点肿",
    "吃咸了脸肿",
    "昨晚哭了眼皮肿",
    "跑完喘不上气",
    "好久没运动了，跑一圈就喘不上气",
    "叫了他一声他没反应过来",
    "练完突然倒了杯水猛喝",
    "他倒了两杯蛋白粉",
    "坐着也能练腹肌吗",
    "走几步就到健身房了",
    # 冲刺 2 二次复核发现的误报
    "他摔了一跤，我叫他不应该再练了",
    "突然叫他一声他都没反应过来",
    "冬天跑步手指失去知觉了",
    "我天生嘴歪",
    "跑完坐着喘口气再练",
    "今天什么都没干，好累",
    # 冲刺 2 第四轮复核：同分句排除、病史与直白说法
    "我睡得太死，谁都叫不醒我",
    "我家孩子每天早上都喊不醒",
    "昨晚练到累倒了，早上我妈都叫不醒我",
    "感冒了喉咙肿",
    "练完二头肌剧痛",
    "我爸去年中风，现在能做什么运动",
    "中风后能跑步吗",
    "心梗后能练吗",
    "以前心梗，现在想减肥",
    # 冲刺 2 第四轮复核（含“说的是别人讲不清楚”）
    "这个视频说话不太清楚，能再解释一下吗",
    "今天练腿练到快倒了，明天还能练吗",
    "我倒是挺想减肥的，就是管不住嘴",
    "这个动作我做的时候身体总往左边歪，怎么纠正",
    "深蹲到底的时候膝盖有点疼过，现在好了",
    "卧推的时候杠铃差点砸到头",
    "我妈说我吃饭时手抖，是不是太饿了",
    "跑步的时候老是岔气，左边肚子疼",
    "这个月体重突然掉了3斤，是不是很厉害",
    "我叫教练帮我看动作，他半天没反应",
    "今天心情不好，不想活动",
    "蛋白粉喝完肚子疼，是不是过敏",
    "我叫我老公起床跑步，叫了三遍都不醒，最后他倒头又睡了",
    "我老公在厨房倒了杯水，叫他半天没反应",
    "我男朋友叫他起来跑步不应，躺在地上玩手机",
    "卷腹的时候脖子疼过一阵",
    "跑步机上跑太快差点摔倒",
    "跳绳跳得我喘不上气，太累了",
    "HIIT做完心跳很快，休息五分钟就好了",
    "硬拉完腰疼，屁股都练麻了",
    "今天跑了十公里，到家直接倒下就睡",
    "今天练腿练到怀疑人生，差点死在健身房",
    "教练让我做了100个波比，人都要没了，喘不过气",
    "这组深蹲做完我直接原地去世",
    "跑完五公里感觉要窒息了",
    "今天臀腿日，练到腿软直接倒了",
    "跑步机上看综艺笑到喘不上气",
    "今天吃撑了，撑得喘不过气",
    # 冲刺 2 第三轮复核发现的误报
    "喝了蜂蜜水脸肿，跑步喘不上气",
    "我叫了外卖半天没反应，倒是快递先到了",
    "叫老公陪我跑步他不答应，突然想自己去了",
    "叫他一起练倒立他不答应",
    "喊他去健身房他没反应，真倒霉",
    "我叫他别吃宵夜他不听也没反应，反倒吃得更多",
    "拍了半天体脂秤没反应，突然想起没电了",
    "早上闹钟叫不醒我，昨晚喝晕了",
    "叫他少喝点他不答应，最后喝晕了",
    "我妈喊我吃饭我没反应，她突然生气了",
    "叫他别摔杠铃他没反应",
    "晕倒过一次，最近想开始减肥",
    "我晕倒过，今天想问问能不能跑步",
    "以前晕倒过，最近腿也疼",
    "以前胸口疼过，最近跑步膝盖也疼",
    "胸口疼过，每次去医院都说没事",
    "跑完坐着喘了半天",
    "做完波比跳躺着喘",
    "组间休息的时候喘得厉害",
    "冲刺的时候一下子跑太快喘不上气",
    "我突然想跑步，跑了5公里喘不上气",
    "突然加大强度，喘不过气",
    "喝了蜂蜜水，早上起来脸肿",
    "中午吃了虾，晚上吃咸了脸肿",
    "跑步时身上痒",
    "吃咸了脸肿，跑步喘不上气",
    "深蹲时身体往右边歪",
    "自拍左脸歪了，修一下",
    "右脸下垂是不是要做提拉",
    "突然发现照片里脸歪了",
    "冲刺完喘不上气，走两步就缓过来了",
    "以前爬一层楼都喘不上气，现在能跑5公里了",
    "最近睡觉时间太少，跑步就喘不上气",
    "不动也胖，一动就喘不上气",
    "有没有坐着也能做的有氧，我一跑就喘不上气",
    "下班走两步去健身房，跑了半小时喘不上气",
    "练完腿直接倒在地上起不来，太累了",
    "哑铃倒在地上砸了个坑",
    "在跑步机上没站稳差点栽倒",
    "滑了一跤栽倒了，膝盖擦破皮",
    "我妈每天早上都喊不醒我",
    "闹钟根本喊不醒我",
    "这叫睡不醒吧，每天都困",
    "我叫了外卖半天没反应",
    "我叫他一起跑步他没回应",
    "喊他去健身房他没反应",
    "看到肥肉突然倒了胃口",
    "一下子倒了半瓶油进去",
    "突然想吃火锅，结果汤倒了",
    "深蹲的时候身体总往右边歪",
    "卧推时右手那边杠铃歪了",
    "硬拉时身子往左边歪怎么办",
    "拍照嘴角歪着笑",
    "我天生嘴角歪一点",
    "嘴歪歪的好可爱",
    "教练讲话不清不楚的，我没听懂",
    "去年晕倒过一次，现在又胖回来了",
    "以前胸口疼过，最近又想开始练胸",
    "以前晕倒过，想再次挑战马拉松",
    "以前胸口疼过还是想练胸",
]


# 已知取舍（spec §33）：宁可多拦的保持拦下，有歧义且缺少上下文的不拦。改动须同时改 spec
TRADEOFFS = [
    ("我不想活得这么累，想好好减肥", "red_flag"),
    ("跑步机突然停了，我头疼得厉害", "emergency"),
    ("我今天在健身房看到有人晕倒了，好吓人", "emergency"),
    ("心跳很快，看到喜欢的人就头晕", "emergency"),
    ("我今天想问以前胸口疼过能不能练", "red_flag"),  # 拿不准是不是既往史时按出现处理
    ("跑完400米喘不上气，快憋死了", "emergency"),
    ("喊他名字也不应", "none"),
    ("她跑完步晕了", "none"),
    ("喘不上气，要死了", "none"),
]


@pytest.mark.parametrize(("text", "expect"), TRADEOFFS)
def test_known_tradeoffs(text: str, expect: str) -> None:
    assert assess(text).level == expect


# 否定说法与以前发生过（spec §22）
NEGATIONS = [
    ("没有胸痛", "none"),
    ("今天不头晕", "none"),
    ("没有心慌也没有胸闷", "none"),
    ("怀孕五个月，运动后未见红", "none"),
    ("胸口不疼了", "none"),
    ("没有心慌，但是胸口还在疼", "emergency"),
    ("胸口不闷但是很疼", "red_flag"),  # 同一分句里另一处没被否定
    ("差点没晕倒", "emergency"),
    ("大太阳下跑步，不出汗了", "emergency"),  # 说法本身带否定字
    ("跑完喘不上气，嘴唇发紫", "emergency"),
    ("爬楼梯胸闷，坐下也不缓解", "emergency"),
    ("我朋友以前晕倒过", "none"),
    ("上周胸口疼过，查了没问题", "none"),
    ("以前胸口疼过，今天又疼了", "red_flag"),  # 再次发生：以前发生过的照常算
    ("以前从来没有过，今天跑步的时候，晕倒过一次", "emergency"),  # 久远的时间只看所在分句
    ("今天想问一下，我以前晕倒过", "none"),  # 前一分句的近期字样不抵消本分句的久远时间
    ("跑完又喘又累，去年晕倒过一次", "none"),  # 复发说法只看本分句与后面的分句
    ("以前晕倒过，最近又犯懒", "none"),
    # 近期字样只在同一分句里才算（冲刺 2 复核 N3）
    ("去年晕倒过一次，最近想开始减肥", "none"),
    ("以前胸口疼过，今天想问问能不能练胸", "none"),
    ("以前心慌过，这周开始跑步了", "none"),
    ("今天晕倒过一次", "emergency"),
    ("刚才胸口疼过一阵", "red_flag"),
    ("晕倒过去了", "emergency"),  # “过去”不是以前发生过
    ("没有胸痛。胸口很闷", "red_flag"),  # 否定只影响所在分句
    ("没头痛，血压190/120", "red_flag"),
    (
        "说没有胸闷其实也胸闷",
        "red_flag",
    ),  # 同一说法出现两处，第一处被否定、第二处没有  # 血压数值不受否定影响；头痛被否定，不升为急症
]


@pytest.mark.parametrize(("text", "expect"), NEGATIONS)
def test_negation_and_past(text: str, expect: str) -> None:
    assert assess(text).level == expect


@pytest.mark.parametrize(("text", "expect"), PARAPHRASES)
def test_paraphrases(text: str, expect: str) -> None:
    assert assess(text).level == expect


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_fitness_talk_not_flagged(text: str) -> None:
    assert assess(text) == Assessment(level="none")


def test_emergency_wins_and_all_rules_listed() -> None:
    result = assess("胸口疼，现在还在疼")
    assert result.level == "emergency"
    assert {"EM-CHEST-ONGOING", "RF-CHEST"} <= set(result.rule_ids)
    assert result.categories == ("cardiovascular",)


def test_combo_within_one_clause_only() -> None:
    assert assess("胸口那里有点疼").level == "red_flag"
    assert assess("胸口有点酸，膝盖疼").level == "none"  # “胸口”与“疼”分在两个分句


UNIFIED = [
    "血压 190／120 ＭＭＨＧ，头痛",
    "胸 口 痛",
    "高压１９０",
    "胸\u3000口\u00a0痛",
    "胸\u200b闷",
    # Unicode 16 新增的“描边数字”：运行时规范化在不同版本下结果不同，统一文字后不再依赖它
    "血压\U0001ccf1\U0001ccf9\U0001ccf0/110，头痛",
    "ＣＨＥＳＴ胸痛",
    "胸口痛\uff61还在疼",  # 半角句号（冲刺 2 复核 N4）
]


def test_unified_text() -> None:
    assert clauses("血压 190／120 mmHg，头痛！") == ["血压190/120mmhg", "头痛"]
    assert assess("血压 190／120 ＭＭＨＧ，头痛").level == "emergency"
    assert assess("胸 口 痛").level == "red_flag"
    assert assess("高压１９０").level == "red_flag"
    assert clauses("胸口痛\uff61还在疼\ufe50头晕") == ["胸口痛", "还在疼", "头晕"]


BLOOD_PRESSURE = [
    ("血压179/109", "none"),
    ("血压180/90", "red_flag"),
    ("血压150/110", "red_flag"),  # 3 级高血压：舒张压 ≥ 110
    ("血压150/120", "red_flag"),
    ("血压190/120", "red_flag"),
    ("血压190/120，头痛", "emergency"),
    ("190/120，头痛", "none"),  # 没有“血压”等字样，不当作血压
    ("低压125", "red_flag"),
    ("血压1900/120", "red_flag"),  # 收缩压位数不合理被忽略，舒张压仍达阈值
    ("血压1900/80", "none"),
]


@pytest.mark.parametrize(("text", "expect"), BLOOD_PRESSURE)
def test_blood_pressure(text: str, expect: str) -> None:
    assert assess(text).level == expect


def test_result_has_no_user_text() -> None:
    result = assess("我爸刚才跑步时说胸痛，然后就倒下了")
    assert set(Assessment.model_fields) == {"level", "rule_ids", "categories"}
    assert result.model_dump_json().isascii()


# ---- 词表校验 ----


def _book() -> dict[str, Any]:
    return {
        "version": "t",
        "review": "pending_review",
        "negation": {"before": ["没有"], "past_after": "过", "recurrence": ["又"]},
        "symptoms": [
            {"id": "chest_pain", "name": "胸痛", "terms": ["胸痛"]},
            {"id": "ongoing", "name": "持续", "combos": [[["还在"], ["疼", "痛"]]]},
        ],
        "rules": [
            {
                "id": "EM-CHEST",
                "level": "emergency",
                "category": "cardiovascular",
                "title": "胸痛持续",
                "all": ["chest_pain", "ongoing"],
            }
        ],
    }


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "terms.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_minimal_book_loads(tmp_path: Path) -> None:
    book = load_terms(_write(tmp_path, _book()))
    assert book.assess("胸痛，现在还在痛").level == "emergency"
    assert book.assess("胸痛").level == "none"


def _symptom(data: dict[str, Any], **changes: Any) -> dict[str, Any]:
    data["symptoms"][0].update(changes)
    return data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["symptoms"].append(dict(d["symptoms"][0])), "编号重复"),
        (lambda d: d["rules"][0]["all"].append("fever"), "不存在的症状"),
        (
            lambda d: d["symptoms"].append({"id": "spare", "name": "多余", "terms": ["x"]}),
            "没有被任何条目使用",
        ),
        (lambda d: _symptom(d, terms=[""]), "terms"),
        (lambda d: _symptom(d, terms=["胸痛，胸闷"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["胸 痛"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["ＢＰ"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["Chest"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=[]), "没有任何说法"),
        (lambda d: _symptom(d, combos=[[["胸口"]]]), "至少两组"),
        (lambda d: d["rules"][0].update(level="red_flag"), "前缀"),
        (lambda d: d["rules"][0].update(all=["chest_pain", "chest_pain", "ongoing"]), "重复引用"),
        (lambda d: d["rules"][0].update(extra=1), "extra"),
        (lambda d: d["negation"].update(before=["没 有"]), "统一后的写法"),
        (lambda d: d["negation"].update(before=[]), "before"),
        (lambda d: d.pop("negation"), "negation"),
    ],
)
def test_bad_book_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = _book()
    mutate(data)
    with pytest.raises(TermLoadError, match=message):
        load_terms(_write(tmp_path, data))


def test_duplicate_yaml_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text("version: a\nversion: b\n", encoding="utf-8")
    with pytest.raises(TermLoadError, match="重复的键"):
        load_terms(path)


def test_bundled_book_is_valid_and_reviewed() -> None:
    book = load_terms(TERMS_PATH)
    assert book.review == "approved"
    assert {r.level for r in book.rules} == {"emergency", "red_flag"}


def test_startup_fails_on_bad_book(tmp_path: Path) -> None:
    data = _book()
    data["rules"][0]["all"] = ["missing"]
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(TermLoadError):
        create_app(
            settings, FixedClock(NOW), log_stream=io.StringIO(), terms_path=_write(tmp_path, data)
        )


def test_bundled_book_round_trips_as_json() -> None:
    """词表能无损转成 JSON（D22 生成前端离线版的前提）。"""
    book = load_terms(TERMS_PATH)
    assert type(book).model_validate_json(json.dumps(book.model_dump(mode="json"))) == book
