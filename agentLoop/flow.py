# flow.py – 100% NetworkX Graph-First (No agentSession)

import networkx as nx
import asyncio
from agentLoop.contextManager import ExecutionContextManager
from agentLoop.agents import AgentRunner
from utils.utils import log_step, log_error
from agentLoop.model_manager import ModelManager
from agentLoop.visualizer import ExecutionVisualizer
from rich.live import Live
from rich.console import Console
from datetime import datetime

class AgentLoop4:
    def __init__(self, multi_mcp, strategy="conservative"):
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.agent_runner = AgentRunner(multi_mcp)

    async def run(self, query, file_manifest, globals_schema, uploaded_files):
        # Phase 1: File Profiling (if files exist)
        file_profiles = {}
        if uploaded_files:
            file_result = await self.agent_runner.run_agent(
                "DistillerAgent",
                {
                    "task": "profile_files",
                    "files": uploaded_files,
                    "instruction": "Profile and summarize each file's structure, columns, content type",
                    "writes": ["file_profiles"]
                }
            )
            if file_result["success"]:
                file_profiles = file_result["output"]

        # Phase 2: Planning with AgentRunner
        plan_result = await self.agent_runner.run_agent(
            "PlannerAgent",
            {
                "original_query": query,
                "planning_strategy": self.strategy,
                "globals_schema": globals_schema,
                "file_manifest": file_manifest,
                "file_profiles": file_profiles
            }
        )

        if not plan_result["success"]:
            raise RuntimeError(f"Planning failed: {plan_result['error']}")

        # Check if plan_graph exists
        if 'plan_graph' not in plan_result['output']:
            raise RuntimeError(f"PlannerAgent output missing 'plan_graph' key. Got: {list(plan_result['output'].keys())}")
        
        plan_graph = plan_result["output"]["plan_graph"]

        try:
            # Phase 3: 100% NetworkX Graph-First Execution
            context = ExecutionContextManager(
                plan_graph,
                session_id=None,
                original_query=query,
                file_manifest=file_manifest
            )
            
            # Add multi_mcp reference
            context.multi_mcp = self.multi_mcp
            
            # Initialize graph with file profiles and globals
            context.set_file_profiles(file_profiles)
            context.plan_graph.graph['globals_schema'].update(globals_schema)

            # Phase 4: Execute DAG with visualization
            await self._execute_dag(context)

            # Phase 5: Return the CONTEXT OBJECT, not summary
            return context

        except Exception as e:
            print(f"❌ ERROR creating ExecutionContextManager: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _execute_dag(self, context):
        """Execute DAG with visualization - DEBUGGING MODE"""
        
        # Get plan_graph structure for visualization
        plan_graph = {
            "nodes": [
                {"id": node_id, **node_data} 
                for node_id, node_data in context.plan_graph.nodes(data=True)
            ],
            "links": [
                {"source": source, "target": target}
                for source, target in context.plan_graph.edges()
            ]
        }
        
        # Create visualizer
        visualizer = ExecutionVisualizer(plan_graph)
        console = Console()
        
        # 🔧 DEBUGGING MODE: No Live display, just regular prints
        max_iterations = 20
        iteration = 0

        while not context.all_done() and iteration < max_iterations:
            iteration += 1
            
            # Show current state
            console.print(visualizer.get_layout())
            
            # Get ready nodes
            ready_steps = context.get_ready_steps()
            
            if not ready_steps:
                # Check for failures
                has_failures = any(
                    context.plan_graph.nodes[n]['status'] == 'failed' 
                    for n in context.plan_graph.nodes
                )
                if has_failures:
                    break
                await asyncio.sleep(0.3)
                continue

            # Mark running
            for step_id in ready_steps:
                visualizer.mark_running(step_id)
                context.mark_running(step_id)
            
            # ✅ EXECUTE AGENTS FOR REAL
            tasks = [self._execute_step(step_id, context) for step_id in ready_steps]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for step_id, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    visualizer.mark_failed(step_id, result)
                    context.mark_failed(step_id, str(result))
                elif result["success"]:
                    visualizer.mark_completed(step_id)
                    await context.mark_done(step_id, result["output"])
                else:
                    visualizer.mark_failed(step_id, result["error"])
                    context.mark_failed(step_id, result["error"])

        # Final state
        console.print(visualizer.get_layout())
        
        if context.all_done():
            console.print("🎉 All tasks completed!")

    async def _execute_step(self, step_id, context):
        """Execute a single step with call_self support"""
        step_data = context.get_step_data(step_id)
        agent_type = step_data["agent"]
        
        # Get inputs from NetworkX graph
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # 🔧 HELPER FUNCTION: Build agent input (consistent for both iterations)
        def build_agent_input(instruction=None, previous_output=None, iteration_context=None):
            if agent_type == "FormatterAgent":
                all_globals = context.plan_graph.graph['globals_schema'].copy()
                return {
                    "step_id": step_id,
                    "agent_prompt": instruction or step_data.get("agent_prompt", step_data["description"]),
                    "reads": step_data.get("reads", []),
                    "writes": step_data.get("writes", []),
                    "inputs": inputs,
                    "all_globals_schema": all_globals,  # ✅ ALWAYS included for FormatterAgent
                    "original_query": context.plan_graph.graph['original_query'],
                    "session_context": {
                        "session_id": context.plan_graph.graph['session_id'],
                        "created_at": context.plan_graph.graph['created_at'],
                        "file_manifest": context.plan_graph.graph['file_manifest']
                    },
                    **({"previous_output": previous_output} if previous_output else {}),
                    **({"iteration_context": iteration_context} if iteration_context else {})
                }
            else:
                return {
                    "step_id": step_id,
                    "agent_prompt": instruction or step_data.get("agent_prompt", step_data["description"]),
                    "reads": step_data.get("reads", []),
                    "writes": step_data.get("writes", []),
                    "inputs": inputs,
                    **({"previous_output": previous_output} if previous_output else {}),
                    **({"iteration_context": iteration_context} if iteration_context else {})
                }

        # --- REFACTORED to support up to 4 calls ---
        max_calls = 4
        iterations_data = []
        final_result = None
        current_result = None
        
        for i in range(max_calls):
            # Determine input for this iteration
            if i == 0:
                agent_input = build_agent_input()
            else:
                log_step(f"↪️ {agent_type} self-call requested. Running iteration {i + 1} for step {step_id}.")
                instruction = current_result["output"].get("next_instruction", "Continue the task")
                previous_output = current_result["output"]
                iteration_context = current_result["output"].get("iteration_context", {})
                agent_input = build_agent_input(instruction, previous_output, iteration_context)

            # Run the agent
            current_result = await self.agent_runner.run_agent(agent_type, agent_input)
            iterations_data.append({"iteration": i + 1, "output": current_result.get("output")})

            if not current_result["success"]:
                final_result = current_result
                break

            output = current_result["output"]
            final_result = current_result

            # Check for code execution between calls
            if output.get("call_self") and context._has_executable_code(output):
                execution_result = await context._auto_execute_code(step_id, output)
                if execution_result.get("status") == "success":
                    execution_data = execution_result.get("result", {})
                    inputs = {**inputs, **execution_data}

            # If agent doesn't want to call self again, stop
            if not output.get("call_self"):
                break
        
        # Store all iteration data in the node for session persistence
        step_node_data = context.get_step_data(step_id)
        step_node_data['iterations'] = iterations_data
        if len(iterations_data) > 1:
            step_node_data['call_self_used'] = True
        if final_result and final_result.get("success"):
            step_node_data['final_iteration_output'] = final_result.get("output")

        return final_result

    async def _handle_failures(self, context):
        """Handle failures via mid-session replanning"""
        # TODO: Implement mid-session replanning with PlannerAgent
        log_error("Mid-session replanning not yet implemented")
