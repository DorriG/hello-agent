## 4. 智能体经典范式构建

### 4.1 ReAct

**ReAct（Reason + Act）：** 核心思想事模仿人类解决问题的方式，将**推理（Reasoning）**与**行动（Acting）**显示地结合起来，形成一个”思考-行动-观察“的循环。

#### 4.1.1 ReAct的工作流程

ReAct认识到思考与行动是相辅相成的。思考指导行动，行动的结果又反过来修正思考。

**实现流程：**

- **Thought：** 分析当前情况、分解任务、指定下一步计划，或者反思上一步结果。
- **Action：** 决定采取的具体动作，通常是调用一个外部工具。
- **Observation：** 这是执行`Action` 后从外部工具返回的结果。Eg：搜索结果的摘要或API的返回值。

智能体不断重复 **Thought -> Action -> Observation** 的循环，将新的观察结果追加到历史记录中，形成一个不断增长的上下文，直到它在`Thought`中认为**已经找到了最终答案** ，然后输出结果。

这个过程形成了一个强大的协同效应：**推理使得行动更具目的性，而行动则为推理提供了事实依据。**

**ReAct的协同循环：**

<img src="https://github.com/DorriG/hello-agent/blob/main/chapter4/pics/1.png" alt="image-20251218211721734" style="zoom:100%;" />

**适用的场景：**

- **需要外部知识的任务**：如查询实时信息（天气、新闻、股价）、搜索专业领域的知识等。
- **需要精确计算的任务**：将数学问题交给计算器工具，避免LLM的计算错误。
- **需要与API交互的任务**：如操作数据库、调用某个服务的API来完成特定功能。

#### 4.1.2 工具的定义与实现

**Tools：** Agent与外部交互的”手和脚“。

**SerApi：** 通过API提供机构化的Google搜索结果，能直接返回”答案摘要框“或精确的知识图谱信息。

安装：

```shell
pip install google-search-results

```

1.  **实现搜索工具的核心逻辑**

   - **名称 (Name)**： 一个简洁、唯一的标识符，供智能体在 `Action` 中调用，例如 `Search`。
   - **描述 (Description)**： 一段清晰的自然语言描述，说明这个工具的用途。**这是整个机制中最关键的部分**，因为大语言模型会依赖这段描述来判断何时使用哪个工具。
   - **执行逻辑 (Execution Logic)**： 真正执行任务的函数或方法。

   ```python
   from serpapi import SerpApiClient
   
   def search(query: str) -> str:
       """
       一个基于SerpApi的实战网页搜索引擎工具。
       它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
       """
       print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
       try:
           api_key = os.getenv("SERPAPI_API_KEY")
           if not api_key:
               return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"
   
           params = {
               "engine": "google",
               "q": query,
               "api_key": api_key,
               "gl": "cn",  # 国家代码
               "hl": "zh-cn", # 语言代码
           }
           
           client = SerpApiClient(params)
           results = client.get_dict()
           
           # 智能解析:优先寻找最直接的答案
           # 首先会检查是否存在 answer_box（Google的答案摘要框）或 knowledge_graph（知识图谱）等信息，如果存在，就直接返回这些最精确的答案。如果不存在，它才会退而求其次，返回前三个常规搜索结果的摘要。
           if "answer_box_list" in results:
               return "\n".join(results["answer_box_list"])
           if "answer_box" in results and "answer" in results["answer_box"]:
               return results["answer_box"]["answer"]
           if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
               return results["knowledge_graph"]["description"]
           if "organic_results" in results and results["organic_results"]:
               # 如果没有直接答案，则返回前三个有机结果的摘要
               snippets = [
                   f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                   for i, res in enumerate(results["organic_results"][:3])
               ]
               return "\n\n".join(snippets)
           
           return f"对不起，没有找到关于 '{query}' 的信息。"
   
       except Exception as e:
           return f"搜索时发生错误: {e}"
   ```

2.  **构建通用的工具执行器**

   当智能体需要使用多种工具时，需要一个统一的管理器来注册和调度这些工具。为此，我们创建一个 `ToolExecutor` 类。

   ```python
   from typing import Dict, Any
   
   class ToolExecutor:
       """
       一个工具执行器，负责管理和执行工具。
       """
       def __init__(self):
           self.tools: Dict[str, Dict[str, Any]] = {}
   
       def registerTool(self, name: str, description: str, func: callable):
           """
           向工具箱中注册一个新工具。
           """
           if name in self.tools:
               print(f"警告:工具 '{name}' 已存在，将被覆盖。")
           self.tools[name] = {"description": description, "func": func}
           print(f"工具 '{name}' 已注册。")
   
       def getTool(self, name: str) -> callable:
           """
           根据名称获取一个工具的执行函数。
           """
           return self.tools.get(name, {}).get("func")
   
       def getAvailableTools(self) -> str:
           """
           获取所有可用工具的格式化描述字符串。
           """
           return "\n".join([
               f"- {name}: {info['description']}" 
               for name, info in self.tools.items()
           ])
   ```

3.  **测试**

   ```python
   # --- 工具初始化与使用示例 ---
   if __name__ == '__main__':
       # 1. 初始化工具执行器
       toolExecutor = ToolExecutor()
   
       # 2. 注册我们的实战搜索工具
       search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
       toolExecutor.registerTool("Search", search_description, search)
       
       # 3. 打印可用的工具
       print("\n--- 可用的工具 ---")
       print(toolExecutor.getAvailableTools())
   
       # 4. 智能体的Action调用，这次我们问一个实时性的问题
       print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
       tool_name = "Search"
       tool_input = "英伟达最新的GPU型号是什么"
   
       tool_function = toolExecutor.getTool(tool_name)
       if tool_function:
           observation = tool_function(tool_input)
           print("--- 观察 (Observation) ---")
           print(observation)
       else:
           print(f"错误:未找到名为 '{tool_name}' 的工具。")
   ```

#### 4.1.3 ReAct智能体的编码实现

1.  **系统提示词设计**

   - **角色定义**： “你是一个有能力调用外部工具的智能助手”，设定了LLM的角色。
   - **工具清单 (`{tools}`)**： 告知LLM它有哪些可用的“手脚”。
   - **格式规约 (`Thought`/`Action`)**： 这是最重要的部分，它强制LLM的输出具有结构性，使我们能通过代码精确解析其意图。
   - **动态上下文 (`{question}`/`{history}`)**： 将用户的原始问题和不断累积的交互历史注入，让LLM基于完整的上下文进行决策。

2.  **核心循环的实现**

   ”格式化提示词->调用LLM->执行动作->整合结果“，直到任务完成或达到最大步数限制。

3.  **输出解析器的实现**

4.  **工具调用与执行**

5.  **观测结果的整合**

6. **运行实例与分析**

~~~python
from typing import List, Dict, Any
# 假设 llm_client.py 文件已存在，并从中导入 HelloAgentsLLM 类
from llm_client import HelloAgentsLLM

# --- 模块 1: 记忆模块 ---

class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        # 初始化一个空列表来存储所有记录
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

# --- 模块 2: Reflection 智能体 ---

# 1. 初始执行提示词
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 2. 反思提示词
REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 3. 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环：反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n{final_code}")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        # 确保能处理生成器可能返回None的情况
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

if __name__ == '__main__':
    # 1. 初始化LLM客户端 (请确保你的 .env 和 llm_client.py 文件配置正确)
    try:
        llm_client = HelloAgentsLLM()
    except Exception as e:
        print(f"初始化LLM客户端时出错: {e}")
        exit()

    # 2. 初始化 Reflection 智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client, max_iterations=2)

    # 3. 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)
~~~

#### 4.1.4 ReAct的特点、局限性与调试

1.  **主要特点**
   - **高可解释性**：ReAct 最大的优点之一就是透明。通过 `Thought` 链，我们可以清晰地看到智能体每一步的“心路历程”——它为什么会选择这个工具，下一步又打算做什么。这对于理解、信任和调试智能体的行为至关重要。
   - **动态规划与纠错能力**：与一次性生成完整计划的范式不同，ReAct 是“走一步，看一步”。它根据每一步从外部世界获得的 `Observation` 来动态调整后续的 `Thought` 和 `Action`。如果上一步的搜索结果不理想，它可以在下一步中修正搜索词，重新尝试。
   - **工具协同能力**：ReAct 范式天然地将大语言模型的推理能力与外部工具的执行能力结合起来。LLM 负责运筹帷幄（规划和推理），工具负责解决具体问题（搜索、计算），二者协同工作，突破了单一 LLM 在知识时效性、计算准确性等方面的固有局限。
2.  **局限性**
   - **对LLM自身能力的强依赖**：ReAct 流程的成功与否，高度依赖于底层 LLM 的综合能力。如果 LLM 的逻辑推理能力、指令遵循能力或格式化输出能力不足，就很容易在 `Thought` 环节产生错误的规划，或者在 `Action` 环节生成不符合格式的指令，导致整个流程中断。
   - **执行效率问题**：由于其循序渐进的特性，完成一个任务通常需要多次调用 LLM。每一次调用都伴随着网络延迟和计算成本。对于需要很多步骤的复杂任务，这种串行的“思考-行动”循环可能会导致较高的总耗时和费用。
   - **提示词的脆弱性**：整个机制的稳定运行建立在一个精心设计的提示词模板之上。模板中的任何微小变动，甚至是用词的差异，都可能影响 LLM 的行为。此外，并非所有模型都能持续稳定地遵循预设的格式，这增加了在实际应用中的不确定性。
   - **可能陷入局部最优**：步进式的决策模式意味着智能体缺乏一个全局的、长远的规划。它可能会因为眼前的 `Observation` 而选择一个看似正确但长远来看并非最优的路径，甚至在某些情况下陷入“原地打转”的循环中。
3.  **调试技巧**
   - **检查完整的提示词**：在每次调用 LLM 之前，将最终格式化好的、包含所有历史记录的完整提示词打印出来。这是追溯 LLM 决策源头的最直接方式。
   - **分析原始输出**：当输出解析失败时（例如，正则表达式没有匹配到 `Action`），务必将 LLM 返回的原始、未经处理的文本打印出来。这能帮助你判断是 LLM 没有遵循格式，还是你的解析逻辑有误。
   - **验证工具的输入与输出**：检查智能体生成的 `tool_input` 是否是工具函数所期望的格式，同时也要确保工具返回的 `observation` 格式是智能体可以理解和处理的。
   - **调整提示词中的示例 (Few-shot Prompting)**：如果模型频繁出错，可以在提示词中加入一两个完整的“Thought-Action-Observation”成功案例，通过示例来引导模型更好地遵循你的指令。
   - **尝试不同的模型或参数**：更换一个能力更强的模型，或者调整 `temperature` 参数（通常设为0以保证输出的确定性），有时能直接解决问题。

### 4.2 Plan-and-Solve

先计划后执行。

#### 4.2.1 工作原理

**核心动机：** 为了解决思维链在处理多步骤、复杂问题时更容易”偏离轨道“的问题。

**工作流程：**

1. **规划阶段 (Planning Phase)**： 首先，智能体会接收用户的完整问题。**将问题分解，并制定出一个清晰、分步骤的行动计划**。这个计划本身就是一次大语言模型的调用产物。
2. **执行阶段 (Solving Phase)**： 在获得完整的计划后，智能体进入执行阶段。它会**严格按照计划中的步骤，逐一执行**。每一步的执行都可能是一次独立的 LLM 调用，或者是对上一步结果的加工处理，直到计划中的所有步骤都完成，最终得出答案。

这种“先谋后动”的策略，使得智能体在处理需要长远规划的复杂任务时，能够保持更高的目标一致性，避免在中间步骤中迷失方向。

**工作流：**

<img src="https://github.com/DorriG/hello-agent/blob/main/chapter4/pics/2.png" alt="image-20251218224251233" style="zoom:1000%;" />

**应用场景：**

- **多步数学应用题**：需要先列出计算步骤，再逐一求解。
- **需要整合多个信息源的报告撰写**：需要先规划好报告结构（引言、数据来源A、数据来源B、总结），再逐一填充内容。
- **代码生成任务**：需要先构思好函数、类和模块的结构，再逐一实现。

#### 4.2.2 规划阶段

为凸显 Plan-and-Solve 范式在结构化推理任务上的优势，通过提示词的设计，完成一个推理任务。

这类任务的特点是，答案无法通过单次查询或计算得出，必须先将问题分解为一系列逻辑连贯的子步骤，然后按顺序求解。这恰好能发挥 Plan-and-Solve “先规划，后执行”的核心能力。

规划阶段的目标是让大语言模型接收原始问题，并输出一个清晰、分步骤的行动计划。这个计划必须是结构化的，以便我们的代码可以轻松解析并逐一执行。因此，我们设计的提示词需要明确地告诉模型它的角色和任务，并给出一个输出格式的范例。

**提示词包含：**

- **角色设定**： “顶级的AI规划专家”，激发模型的专业能力。
- **任务描述**： 清晰地定义了“分解问题”的目标。
- **格式约束**： 强制要求输出为一个 Python 列表格式的字符串，这极大地简化了后续代码的解析工作，使其比解析自然语言更稳定、更可靠。

~~~python
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
~~~

#### 4.2.3 执行器与状态管理

在规划器 (`Planner`) 生成了清晰的行动蓝图后，我们就需要一个执行器 (`Executor`) 来逐一完成计划中的任务。执行器不仅负责调用大语言模型来解决每个子问题，还承担着一个至关重要的角色：**状态管理**。它必须记录每一步的执行结果，并将其作为上下文提供给后续步骤，确保信息在整个任务链条中顺畅流动

执行器的提示词与规划器不同。它的目标不是分解问题，而是**在已有上下文的基础上，专注解决当前这一个步骤**。因此，提示词需要包含以下关键信息：

- **原始问题**： 确保模型始终了解最终目标。
- **完整计划**： 让模型了解当前步骤在整个任务中的位置。
- **历史步骤与结果**： 提供至今为止已经完成的工作，作为当前步骤的直接输入。
- **当前步骤**： 明确指示模型现在需要解决哪一个具体任务。

```python
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""
```

#### 4.2.4 运行实例与分析

~~~python
import os
import ast
from llm_client import HelloAgentsLLM
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量，处理文件不存在异常
try:
    load_dotenv()
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")

# --- 1. LLM客户端定义 ---
# 假设你已经有llm_client.py文件，里面定义了HelloAgentsLLM类

# --- 2. 规划器 (Planner) 定义 ---
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")
        
        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

# --- 3. 执行器 (Executor) 定义 ---
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        final_answer = ""
        
        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "无", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            
            response_text = self.llm_client.think(messages=messages) or ""
            
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")
            
        return final_answer

# --- 4. 智能体 (Agent) 整合 ---
class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

# --- 5. 主函数入口 ---
if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)
~~~

### 4.3 Reflection

**核心思想：**正是为智能体引入一种**事后（post-hoc）的自我校正循环**，使其能够像人类一样，审视自己的工作，发现不足，并进行迭代优化。

#### 4.3.1 核心思想

**工作流程：** **执行 -> 反思 -> 优化**。

1. **执行 (Execution)**：首先，智能体使用我们熟悉的方法（如 ReAct 或 Plan-and-Solve）尝试完成任务，生成一个初步的解决方案或行动轨迹。这可以看作是“初稿”。
2. **反思 (Reflection)：** 智能体进入反思阶段。它会调用一个独立的、或者带有特殊提示词的大语言模型实例，来扮演一个“评审员”的角色。这个“评审员”会审视第一步生成的“初稿”，并从多个维度进行评估，例如：
   - **事实性错误**：是否存在与常识或已知事实相悖的内容？
   - **逻辑漏洞**：推理过程是否存在不连贯或矛盾之处？
   - **效率问题**：是否有更直接、更简洁的路径来完成任务？
   - **遗漏信息**：是否忽略了问题的某些关键约束或方面？ 根据评估，它会生成一段结构化的**反馈 (Feedback)**，指出具体的问题所在和改进建议。
3. **优化 (Refinement)**：最后，智能体将“初稿”和“反馈”作为新的上下文，再次调用大语言模型，要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。

<img src="https://github.com/DorriG/hello-agent/blob/main/chapter4/pics/3.png" alt="image-20251218225257644" style="zoom:100%;" />

#### 4.3.2 案例设定于记忆模块设计

为了在实战中体现 Reflection 机制，我们将引入记忆管理机制，因为reflection通常对应着信息的存储和提取，如果上下文足够长的情况，想让“评审员”直接获取所有的信息然后进行反思往往会传入很多冗余信息。这一步实践我们主要完成**代码生成与迭代优化**。

Reflection 的核心在于迭代，而迭代的前提是能够记住之前的尝试和获得的反馈。因此，一个“短期记忆”模块是实现该范式的必需品。这个记忆模块将负责存储每一次“执行-反思”循环的完整轨迹。

#### 4.3.3 编码实现

1.  **提示词的设计**

   - **初始执行提示词 (Execution Prompt)** ：这是智能体首次尝试解决问题的提示词，内容相对直接，只要求模型完成指定任务。

     ```python
     INITIAL_PROMPT_TEMPLATE = """
     你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
     你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
     
     要求: {task}
     
     请直接输出代码，不要包含任何额外的解释。
     """
     ```

   - **反思提示词 (Reflection Prompt)** ：这个提示词是 Reflection 机制的灵魂。它指示模型扮演“代码评审员”的角色，对上一轮生成的代码进行批判性分析，并提供具体的、可操作的反馈。

     ~~~python
     REFLECT_PROMPT_TEMPLATE = """
     你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
     你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。
     
     # 原始任务:
     {task}
     
     # 待审查的代码:
     ```python
     {code}
     ```
     
     请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
     如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
     如果代码在算法层面已经达到最优，才能回答“无需改进”。
     
     请直接输出你的反馈，不要包含任何额外的解释。
     """
     ~~~

   - **优化提示词 (Refinement Prompt)** ：当收到反馈后，这个提示词将引导模型根据反馈内容，对原有代码进行修正和优化。

     ```python
     REFINE_PROMPT_TEMPLATE = """
     你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。
     
     # 原始任务:
     {task}
     
     # 你上一轮尝试的代码:
     {last_code_attempt}
     评审员的反馈：
     {feedback}
     
     请根据评审员的反馈，生成一个优化后的新版本代码。
     你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
     请直接输出优化后的代码，不要包含任何额外的解释。
     """
     ```

2.  **智能体封装与实现**

~~~python
from typing import List, Dict, Any
# 假设 llm_client.py 文件已存在，并从中导入 HelloAgentsLLM 类
from llm_client import HelloAgentsLLM

# --- 模块 1: 记忆模块 ---

class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        # 初始化一个空列表来存储所有记录
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

# --- 模块 2: Reflection 智能体 ---

# 1. 初始执行提示词
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 2. 反思提示词
REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 3. 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环：反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n{final_code}")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        # 确保能处理生成器可能返回None的情况
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text

if __name__ == '__main__':
    # 1. 初始化LLM客户端 (请确保你的 .env 和 llm_client.py 文件配置正确)
    try:
        llm_client = HelloAgentsLLM()
    except Exception as e:
        print(f"初始化LLM客户端时出错: {e}")
        exit()

    # 2. 初始化 Reflection 智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client, max_iterations=2)

    # 3. 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)
~~~

### 4.4 作业

**作业一：** 本章介绍了三种经典的智能体范式:`ReAct`、`Plan-and-Solve` 和 `Reflection`。请分析:

- 这三种范式在"思考"与"行动"的组织方式上有什么本质区别？
- 如果要设计一个"智能家居控制助手"（需要控制灯光、空调、窗帘等多个设备，并根据用户习惯自动调节），你会选择哪种范式作为基础架构？为什么？
- 是否可以将这三种范式进行组合使用？若可以，请尝试设计一个混合范式的智能体架构，并说明其适用场景。

> **Q1:**
>
> **ReAct：** 思考是**局部的、即时的** ；行动是**探索性的** ；不追求全局最优，只追求“此刻合理”
>
> **Plan-and-Solve：** 思考是**全局的、前置的** ；行动是**执行性的** ； 假设环境在执行期间“足够稳定”
>
> **Reflect：** 思考不直接驱动当前行动；而是**塑造未来的行动方式** ； 是“学习机制”，不是“控制机制”
>
> **Q2：**
>
> 以 Plan-and-Solve 为主，Reflection 为核心增强，ReAct 作为局部辅助。
>
> 因为智能家居本质是：多目标约束系统（舒适度、能耗、用户偏好）；行为是**可预演的**；环境变化节奏相对慢。根据用户习惯自动调节，习惯需要长期统计和反思总结共同得到。
>
> **Q3：**
>
> 可以。Planning Layer（Plan-and-Solve）根据当前状态 + 用户模型生成行为计划，输出设备控制序列。Execution Layer（ReAct）处理突发情况，快速纠偏。Reflection Layer（Reflection）分析用户是否频繁手动覆盖系统决策，哪些自动策略被“否定”。

**作业二：** 在4.2节的 `ReAct` 实现中，我们使用了正则表达式来解析大语言模型的输出（如 `Thought` 和 `Action`）。请思考:

- 当前的解析方法存在哪些潜在的脆弱性？在什么情况下可能会失败？
- 除了正则表达式，还有哪些更鲁棒的输出解析方案？
- 尝试修改本章的代码，使用一种更可靠的输出格式，并对比两种方案的优缺点

> **Q1:**
>
> 1. 对 LLM 输出“表面形态”高度敏感
> 2. 正则解析 = 把“语义结构”当成“字符串形状”，无法扩展
>
> **Q2：**
>
> 1. JSON / YAML 结构化输出（结构稳定、语义清晰、易扩展；prompt设计严格）
> 2. XML / 标签化 DSL
> 3. 函数调用（Function Calling / Tool Calling）
>
> action才是最重要的，换言之数据很重要。
>
> **Q3：**
>
> ```python
> import json
> from llm_client import HelloAgentsLLM
> from tools import ToolExecutor, search
> 
> REACT_JSON_PROMPT_TEMPLATE = """
> 你是一个可以调用外部工具的智能体。
> 
> 你必须【只返回一个合法 JSON 对象】，不要输出任何解释性文字、markdown 或多余内容。
> 
> 返回格式必须严格如下：
> 
> {
>   "thought": "你的思考，用于分析问题和决定下一步行动",
>   "action": {
>     "type": "tool" | "finish",
>     "name": "工具名称（当 type=tool 时填写）",
>     "input": "工具输入（当 type=tool 时填写）",
>     "answer": "最终答案（当 type=finish 时填写）"
>   }
> }
> 
> 可用工具如下：
> {tools}
> 
> Question:
> {question}
> 
> History:
> {history}
> """
> 
> class JSONReActAgent:
>     def __init__(
>         self,
>         llm_client: HelloAgentsLLM,
>         tool_executor: ToolExecutor,
>         max_steps: int = 5
>     ):
>         self.llm_client = llm_client
>         self.tool_executor = tool_executor
>         self.max_steps = max_steps
>         self.history = []
> 
>     def run(self, question: str):
>         self.history = []
>         step = 0
> 
>         while step < self.max_steps:
>             step += 1
>             print(f"\n====== Step {step} ======")
> 
>             prompt = self._build_prompt(question)
>             messages = [{"role": "user", "content": prompt}]
> 
>             response_text = self.llm_client.think(messages=messages)
>             if not response_text:
>                 print("❌ LLM 返回为空，终止执行")
>                 break
> 
>             thought, action = self._parse_llm_output(response_text)
> 
>             if thought:
>                 print(f"🤔 Thought: {thought}")
> 
>             if not action:
>                 print("❌ 无法解析 Action，终止执行")
>                 break
> 
>             if action["type"] == "finish":
>                 answer = action.get("answer", "")
>                 print(f"\n🎉 Final Answer:\n{answer}")
>                 return answer
> 
>             if action["type"] == "tool":
>                 tool_name = action.get("name")
>                 tool_input = action.get("input")
> 
>                 print(f"🔧 Action: {tool_name}[{tool_input}]")
> 
>                 tool_func = self.tool_executor.getTool(tool_name)
>                 if not tool_func:
>                     observation = f"工具 {tool_name} 不存在"
>                 else:
>                     observation = tool_func(tool_input)
> 
>                 print(f"👀 Observation: {observation}")
> 
>                 self._update_history(thought, action, observation)
>                 continue
> 
>             print("⚠️ 未知的 action type，终止执行")
>             break
> 
>         print("⏹ 已达到最大步数，任务未完成")
>         return None
> 
>     # -------------------------
>     # Prompt 构建
>     # -------------------------
>     def _build_prompt(self, question: str) -> str:
>         tools_desc = self.tool_executor.getAvailableTools()
>         history_text = "\n".join(self.history)
> 
>         return REACT_JSON_PROMPT_TEMPLATE.format(
>             tools=tools_desc,
>             question=question,
>             history=history_text
>         )
> 
>     # -------------------------
>     # JSON 解析（关键）
>     # -------------------------
>     def _parse_llm_output(self, text: str):
>         try:
>             data = json.loads(text)
>             thought = data.get("thought")
>             action = data.get("action")
>             return thought, action
>         except json.JSONDecodeError as e:
>             print("❌ JSON 解析失败")
>             print(text)
>             return None, None
> 
>     # -------------------------
>     # History 维护
>     # -------------------------
>     def _update_history(self, thought, action, observation):
>         self.history.append(f"Thought: {thought}")
>         self.history.append(f"Action: {action}")
>         self.history.append(f"Observation: {observation}")
> ```

**作业三：** 工具调用是现代智能体的核心能力之一。基于4.2.2节的 `ToolExecutor` 设计，请完成以下扩展实践:

- 为 `ReAct` 智能体添加一个"计算器"工具，使其能够处理复杂的数学计算问题（如"计算 `(123 + 456) × 789/ 12 = ?` 的结果"）
- 设计并实现一个"工具选择失败"的处理机制:当智能体多次调用错误的工具或提供错误的参数时，系统应该如何引导它纠正？
- 思考:如果可调用工具的数量增加到5050个甚至100100个，当前的工具描述方式是否还能有效工作？在可调用工具数量随业务需求显著增加时，从工程角度如何优化工具的组织和检索机制？

> **Q1:**
>
> ```python
> import ast
> import operator
> 
> _ALLOWED_OPERATORS = {
>     ast.Add: operator.add,
>     ast.Sub: operator.sub,
>     ast.Mult: operator.mul,
>     ast.Div: operator.truediv,
>     ast.Pow: operator.pow,
>     ast.USub: operator.neg,
> }
> 
> def _eval(node):
>     if isinstance(node, ast.Num):  # Python <3.8
>         return node.n
>     if isinstance(node, ast.Constant):  # Python >=3.8
>         return node.value
>     if isinstance(node, ast.BinOp):
>         left = _eval(node.left)
>         right = _eval(node.right)
>         op = _ALLOWED_OPERATORS[type(node.op)]
>         return op(left, right)
>     if isinstance(node, ast.UnaryOp):
>         return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
>     raise ValueError("不支持的表达式")
> 
> def calculator(expression: str) -> str:
>     try:
>         tree = ast.parse(expression, mode="eval")
>         result = _eval(tree.body)
>         return str(result)
>     except Exception as e:
>         return f"计算错误: {e}"
> 
>     tool_executor.registerTool(
>     name="Calculator",
>     description="用于精确计算数学表达式，如 (123 + 456) * 789 / 12",
>     func=calculator
> )
> ```
>
> **Q2:**
>
> ```python
> def _handle_tool_call(self, action):
>     tool = action["name"]
>     inp = action["input"]
> 
>     if tool not in self.tool_executor.tools:
>         return self._tool_error(
>             tool,
>             f"工具 {tool} 不存在，请从可用工具中选择"
>         )
> 
>     result = self.tool_executor.getTool(tool)(inp)
> 
>     if "错误" in str(result):
>         return self._tool_error(
>             tool,
>             f"工具执行失败，原因: {result}"
>         )
> 
>     return result
> 
> def _tool_error(self, tool_name, message):
>     self.tool_error_count[tool_name] = self.tool_error_count.get(tool_name, 0) + 1
> 
>     if self.tool_error_count[tool_name] >= 3:
>         return (
>             "你多次错误使用该工具。\n"
>             "请重新思考问题，或选择其他工具，"
>             "或直接给出最终答案。"
>         )
> 
>     return message
> ```
>
> **Q3:**
>
> 全放prompt，解析的时候token会爆炸。
>
> 解决方案：
>
> 1.  把tools也用向量化检索，工具本身看作是文档。
> 2.  建立分层工具体系，哪些任务适合哪一类工具。

**作业四：** `Plan-and-Solve` 范式将任务分解为"规划"和"执行"两个阶段。请深入分析:

- 在4.3节的实现中，规划阶段生成的计划是"静态"的（一次性生成，不可修改）。如果在执行过程中发现某个步骤无法完成或结果不符合预期，应该如何设计一个"动态重规划"机制？
- 对比 `Plan-and-Solve` 与 `ReAct`:在处理"预订一次从北京到上海的商务旅行（包括机票、酒店、租车）"这样的任务时，哪种范式更合适？为什么？
- 尝试设计一个"分层规划"系统:先生成高层次的抽象计划，然后针对每个高层步骤再生成详细的子计划。这种设计有什么优势？

**作业五：** `Reflection` 机制通过"执行-反思-优化"循环来提升输出质量。请思考:

- 在4.4节的代码生成案例中，不同阶段使用的是同一个模型。如果使用两个不同的模型（例如，用一个更强大的模型来做反思，用一个更快的模型来做执行），会带来什么影响？
- `Reflection` 机制的终止条件是"反馈中包含**无需改进**"或"达到最大迭代次数"。这种设计是否合理？能否设计一个更智能的终止条件？
- 假设你要搭建一个"学术论文写作助手"，它能够生成初稿并不断优化论文内容。请设计一个多维度的Reflection机制，从段落逻辑性、方法创新性、语言表达、引用规范等多个角度进行反思和改进。

**作业六：** 提示词工程是影响智能体最终效果的关键技术。本章展示了多个精心设计的提示词模板。请分析:

- 对比4.2.3节的 `ReAct` 提示词和4.3.2节的 `Plan-and-Solve` 提示词，它们显然存在结构设计上的明显不同，这些差异是如何服务于各自范式的核心逻辑的？
- 在4.4.3节的 `Reflection` 提示词中，我们使用了"你是一位极其严格的代码评审专家"这样的角色设定。尝试修改这个角色设定（如改为"你是一位注重代码可读性的开源项目维护者"），观察输出结果的变化，并总结角色设定对智能体行为的影响。
- 在提示词中加入 `few-shot` 示例往往能显著提升模型对特定格式的遵循能力。请为本章的某个智能体尝试添加 `few-shot` 示例，并对比其效果。

**作业七：** 某电商初创公司现在希望使用"客服智能体"来代替真人客服实现降本增效，它需要具备以下功能:

a. 理解用户的退款申请理由

b. 查询用户的订单信息和物流状态

c. 根据公司政策智能地判断是否应该批准退款

d. 生成一封得体的回复邮件并发送至用户邮箱

e. 如果判断决策存在一定争议（自我置信度低于阈值），能够进行自我反思并给出更审慎的建议

此时作为该产品的负责人:

- 你会选择本章的哪种范式（或哪些范式的组合）作为系统的核心架构？
- 这个系统需要哪些工具？请列出至少3个工具及其功能描述。
- 如何设计提示词来确保智能体的决策既符合公司利益，又能保持对用户的友好态度？

- 这个产品上线后可能面临哪些风险和挑战？如何通过技术手段来降低这些风险？
