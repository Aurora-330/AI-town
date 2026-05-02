"""批量NPC对话生成器"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Optional

# 添加HelloAgents到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HelloAgents'))

from hello_agents import HelloAgentsLLM
from config import settings as _settings  # 触发 backend/.env 加载
from agents import NPC_ROLES

class NPCBatchGenerator:
    """批量生成NPC对话的生成器
    
    核心思路: 一次LLM调用生成所有NPC的对话,降低API成本和延迟
    """
    
    def __init__(self):
        """初始化批量生成器"""
        print("🎨 正在初始化批量对话生成器...")
        
        try:
            self.llm = HelloAgentsLLM()
            self.enabled = True
            print("✅ 批量生成器初始化成功")
        except Exception as e:
            print(f"❌ 批量生成器初始化失败: {e}")
            print("⚠️  将使用预设对话模式")
            self.llm = None
            self.enabled = False
        
        self.npc_configs = NPC_ROLES
        
        # 预设对话库(当LLM不可用时使用)
        self.preset_dialogues = {
            "morning": {
                "风泠": "早。我先把昨晚留下的记录补完,免得细节又散掉了。",
                "郁米": "新的一天先从整理心情开始,这样说话才不会把疲惫带给别人。",
                "顾辰": "先把今天的目标拆开,这样下午之前就知道哪里会卡住。"
            },
            "noon": {
                "风泠": "上午的访客记录已经整理完了,有两条时间线对不上,下午得再核一次。",
                "郁米": "中午适合把情绪放慢一点,很多着急说出口的话其实可以换个方式。",
                "顾辰": "方案主线已经清了,午饭前再把风险点压缩成三条就够了。"
            },
            "afternoon": {
                "风泠": "下午适合回头查遗漏,很多真正关键的东西都藏在不起眼的角落里。",
                "郁米": "有人下午会比上午更容易烦躁,先被理解,事情才谈得下去。",
                "顾辰": "如果目标不变,这版方案今晚前可以收敛;如果目标变了,就得重拆。"
            },
            "evening": {
                "风泠": "今天的记录差不多归档完了,剩下那点疑点,明天再追。",
                "郁米": "晚上适合把没说清的情绪轻轻放下,别让它们跟着你回去。",
                "顾辰": "今天先到这里,路线已经够清楚了,明天直接推进执行。"
            }
        }
    
    def generate_batch_dialogues(self, context: Optional[str] = None) -> Dict[str, str]:
        """批量生成所有NPC的对话
        
        Args:
            context: 场景上下文(如"上午工作时间"、"午餐时间"等)
        
        Returns:
            Dict[str, str]: NPC名称到对话内容的映射
        """
        if not self.enabled or self.llm is None:
            # 使用预设对话
            return self._get_preset_dialogues()
        
        try:
            # 构建批量生成提示词
            prompt = self._build_batch_prompt(context)

            # 一次LLM调用生成所有对话
            # 使用invoke方法而不是chat方法
            response = self.llm.invoke([
                {"role": "system", "content": "你是一个游戏NPC对话生成器,擅长创作自然真实的办公室对话。"},
                {"role": "user", "content": prompt}
            ])

            # 解析JSON响应
            dialogues = self._parse_response(response)

            if dialogues:
                print(f"✅ 批量生成成功: {len(dialogues)}个NPC对话")
                return dialogues
            else:
                print("⚠️  解析失败,使用预设对话")
                return self._get_preset_dialogues()

        except Exception as e:
            print(f"❌ 批量生成失败: {e}")
            return self._get_preset_dialogues()
    
    def _build_batch_prompt(self, context: Optional[str] = None) -> str:
        """构建批量生成提示词"""
        # 根据时间自动推断场景
        if context is None:
            context = self._get_current_context()
        
        # 构建NPC描述
        npc_descriptions = []
        for name, cfg in self.npc_configs.items():
            desc = f"- {name}({cfg['title']}): 在{cfg['location']}{cfg['activity']},性格{cfg['personality']}"
            npc_descriptions.append(desc)
        
        npc_desc_text = "\n".join(npc_descriptions)
        
        prompt = f"""请为Datawhale办公室的3个NPC生成当前的对话或行为描述。

【场景】{context}

【NPC信息】
{npc_desc_text}

【生成要求】
1. 每个NPC生成1句话(20-40字)
2. 内容要符合角色设定、当前活动和场景氛围
3. 可以是自言自语、工作状态描述、或简单的思考
4. 要自然真实,像真实的办公室同事
5. 可以体现一些个性化特点和情绪
6. **必须严格按照JSON格式返回**

【输出格式】(严格遵守)
{{"风泠": "...", "郁米": "...", "顾辰": "..."}}

【示例输出】
{{"风泠": "这个bug真是见鬼了,已经调试两小时了...", "郁米": "嗯,这个功能的优先级需要重新评估一下。", "顾辰": "这杯咖啡的拉花真不错,灵感来了!"}}

请生成(只返回JSON,不要其他内容):
"""
        return prompt
    
    def _parse_response(self, response: str) -> Optional[Dict[str, str]]:
        """解析LLM响应"""
        try:
            # 尝试直接解析JSON
            dialogues = json.loads(response)
            
            # 验证格式
            if isinstance(dialogues, dict) and all(name in dialogues for name in self.npc_configs.keys()):
                return dialogues
            else:
                print(f"⚠️  JSON格式不正确: {dialogues}")
                return None
                
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            try:
                # 查找第一个{和最后一个}
                start = response.find('{')
                end = response.rfind('}') + 1
                
                if start != -1 and end > start:
                    json_str = response[start:end]
                    dialogues = json.loads(json_str)
                    
                    if isinstance(dialogues, dict):
                        return dialogues
            except:
                pass
            
            print(f"⚠️  无法解析响应: {response[:100]}...")
            return None
    
    def _get_current_context(self) -> str:
        """根据当前时间推断场景上下文"""
        hour = datetime.now().hour
        
        if 6 <= hour < 9:
            return "清晨时分,大家陆续到达办公室,准备开始新的一天"
        elif 9 <= hour < 12:
            return "上午工作时间,大家都在专注工作,办公室氛围专注而忙碌"
        elif 12 <= hour < 14:
            return "午餐时间,大家在休息放松,聊聊天或者看看手机"
        elif 14 <= hour < 17:
            return "下午工作时间,继续推进项目,偶尔需要喝杯咖啡提神"
        elif 17 <= hour < 19:
            return "傍晚时分,准备收尾今天的工作,整理明天的计划"
        else:
            return "夜晚时分,办公室安静下来,偶尔还有人在加班"
    
    def _get_preset_dialogues(self) -> Dict[str, str]:
        """获取预设对话(根据时间)"""
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 14:
            period = "noon"
        elif 14 <= hour < 18:
            period = "afternoon"
        else:
            period = "evening"
        
        return self.preset_dialogues.get(period, self.preset_dialogues["morning"])

# 全局单例
_batch_generator = None

def get_batch_generator() -> NPCBatchGenerator:
    """获取批量生成器单例"""
    global _batch_generator
    if _batch_generator is None:
        _batch_generator = NPCBatchGenerator()
    return _batch_generator
