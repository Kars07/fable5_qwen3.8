"""
A taskset is the collection and loader for the work to evaluate or train on. Each task combines a serializable TaskData row (prompt, files, references, resource requirements) 
with its task class’s behavior (lifecycle hooks, tools, metrics, and rewards). 
The taskset’s load() method constructs those objects and declares their task/config types through Taskset[TaskT, ConfigT].

"""
