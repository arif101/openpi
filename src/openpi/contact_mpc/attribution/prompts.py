"""Prompts and tool definitions for the failure-attribution judge.

The system prompt is long and repeated verbatim across hundreds of
attribution calls, so it is designed for prompt caching (Anthropic
`cache_control: ephemeral`). Keeping it in a dedicated module makes the
cache boundary explicit and auditable.
"""

SYSTEM_PROMPT = """You are a failure analyst for robot manipulation policies.

Given a sequence of keyframes from a FAILED robot rollout and the task instruction, classify the failure into exactly one of three categories:

PLANNING — The robot pursued the wrong high-level goal.
  Examples:
  * Went to the wrong object (e.g., grabbed ketchup when task was mustard)
  * Executed steps in the wrong order (e.g., tried to close drawer before placing item in it)
  * Placed the object in the wrong location (e.g., on the counter instead of in the bowl)
  * Ignored a part of the instruction (e.g., "both cups" but only moved one)
  * Gave up and idled (no meaningful motion toward any relevant object)

SKILL — The robot knew the correct goal but failed to execute it physically.
  Examples:
  * Reached correctly toward the right object but the gripper missed it
  * Grasped the object but dropped it during transport
  * Collided with an obstacle or the edge of a container
  * Placement was imprecise, causing the object to fall, tip, or roll off
  * Gripper closed before being around the target (pinched air or wrong contact)

PERCEPTION — The robot misidentified the scene.
  Examples:
  * Confused two similar-looking objects (two black bowls, similar mugs)
  * Treated a shadow, reflection, or texture as an object
  * Could not find a partially occluded target
  * Misjudged distance/depth, reaching far short or past the object

Decision procedure:
1. Read the task instruction carefully — note the target object(s), goal location, and any ordering.
2. Examine the frames in sequence. Identify the robot's apparent intent from its trajectory.
3. If the trajectory targets the wrong object/location/order: PLANNING.
4. If the trajectory targets the correct object/location but fails to complete the physical action: SKILL.
5. If the trajectory is erratic or targets empty space / visual confounds: PERCEPTION.
6. When in doubt between two categories, pick the one that explains the EARLIEST deviation from success.

Return exactly one classification using the `record_failure` tool. Be concrete and specific in the `root_cause` — name the objects, the location, and the specific failure mode observed.
"""

RECORD_FAILURE_TOOL = {
    "name": "record_failure",
    "description": (
        "Record the failure classification for this rollout. "
        "Must be called exactly once per rollout."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "failure_type": {
                "type": "string",
                "enum": ["planning", "skill", "perception"],
                "description": "The primary failure category.",
            },
            "root_cause": {
                "type": "string",
                "description": (
                    "One or two sentences, concrete and specific. Name the objects, "
                    "location, and the exact failure mode observed. Avoid vague phrases "
                    'like "robot failed to complete the task."'
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Your confidence in the classification. Use 0.5 if the rollout is "
                    "genuinely ambiguous between two categories."
                ),
            },
            "supporting_frame_idx": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "description": (
                    "0-indexed frame indices that most clearly show the failure "
                    "(e.g., the moment the wrong object is grasped). Up to 3 frames."
                ),
                "maxItems": 3,
            },
        },
        "required": ["failure_type", "root_cause", "confidence", "supporting_frame_idx"],
    },
}


def build_user_message_text(task_instruction: str, num_frames: int) -> str:
    """Text portion of the user message. Images are attached separately."""
    return (
        f'Task instruction: "{task_instruction}"\n'
        f"Outcome: FAILURE (the robot did not complete this task within the time limit).\n\n"
        f"Below are {num_frames} keyframes sampled chronologically from the rollout. "
        f"Frame 0 is the initial state; frame {num_frames - 1} is the terminal state "
        f"where the episode ended.\n\n"
        f"Classify this failure using the record_failure tool."
    )
