"""Status and story route handlers.

Provides endpoints for sprint status, story listing, and epic details.
"""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


async def get_status(request: Request) -> JSONResponse:
    """GET /api/status - Return sprint status.

    Returns current sprint state from sprint-status.yaml including:
    - Current phase
    - Active story
    - Overall progress
    """
    server = request.app.state.server

    try:
        status = server.get_sprint_status()
        return JSONResponse(status)
    except Exception as e:
        logger.exception("Failed to get sprint status")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_stories(request: Request) -> JSONResponse:
    """GET /api/stories - Return story list with phases.

    Returns hierarchical structure:
    - Epics with metadata
    - Stories within each epic
    - Workflow phases for each story
    """
    server = request.app.state.server

    try:
        stories = server.get_stories()
        return JSONResponse(stories)
    except Exception as e:
        logger.exception("Failed to get stories")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_epic_details(request: Request) -> JSONResponse:
    """GET /api/epics/{epic_id} - Return epic details with full content.

    Returns epic content from epics.md or sharded epic file.
    Includes frontmatter metadata and markdown content.
    """
    server = request.app.state.server
    epic_id = request.path_params.get("epic_id")

    try:
        details = server.get_epic_details(epic_id)
        if details is None:
            return JSONResponse(
                {"error": f"Epic not found: {epic_id}"},
                status_code=404,
            )
        return JSONResponse(details)
    except Exception as e:
        logger.exception("Failed to get epic details")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_story_in_epic(request: Request) -> JSONResponse:
    """GET /api/epics/{epic_id}/stories/{story_id} - Return story content from epic.

    Extracts and returns only the specified story's content from within the epic file.
    Supports both sharded and non-sharded epic files.
    """
    server = request.app.state.server
    epic_id = request.path_params.get("epic_id")
    story_id = request.path_params.get("story_id")

    try:
        story = server.get_story_in_epic(epic_id, story_id)
        if story is None:
            return JSONResponse(
                {"error": f"Story {story_id} not found in epic {epic_id}"},
                status_code=404,
            )
        return JSONResponse(story)
    except Exception as e:
        logger.exception("Failed to get story in epic")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_epic_metrics(request: Request) -> JSONResponse:
    """GET /api/epics/{epic_id}/metrics - Return aggregated benchmark metrics for epic.

    Aggregates all benchmark files for stories in this epic:
    - Total duration across all workflows
    - Per-workflow breakdown (create-story, dev-story, validate, code-review)
    - Story count and completion stats
    """
    server = request.app.state.server
    epic_id = request.path_params.get("epic_id")

    try:
        metrics = server.get_epic_metrics(epic_id)
        if metrics is None:
            return JSONResponse(
                {"error": f"No metrics found for epic: {epic_id}"},
                status_code=404,
            )
        return JSONResponse(metrics)
    except Exception as e:
        logger.exception("Failed to get epic metrics")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_state(request: Request) -> JSONResponse:
    """GET /api/state - Return current execution state from state.yaml.

    Returns the authoritative execution position maintained by the loop:
    - has_position: Whether state has a recorded position (not necessarily actively running)
    - current_epic: Current epic ID (int or str like "testarch")
    - current_story: Current story ID (e.g., "22.3")
    - current_phase: Current phase in snake_case (e.g., "dev_story")
    - phase_started_at: ISO timestamp of when current phase started (from run log)
    - completed_stories: List of completed story IDs (always a list, may be empty)
    - completed_epics: List of completed epic IDs (always a list, may be empty)
    """
    server = request.app.state.server

    try:
        state = server.get_current_state()
        if state is None:
            return JSONResponse({"has_position": False})

        # Get phase start time from run log
        phase_started_at = server.get_phase_started_at()

        return JSONResponse(
            {
                "has_position": state.current_phase is not None,
                "current_epic": state.current_epic,
                "current_story": state.current_story,
                "current_phase": state.current_phase.value if state.current_phase else None,
                "phase_started_at": phase_started_at.isoformat() if phase_started_at else None,
                "completed_stories": state.completed_stories,
                "completed_epics": state.completed_epics,
            }
        )
    except Exception as e:
        logger.exception("Failed to get execution state")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_version(request: Request) -> JSONResponse:
    """GET /api/version - Return bmad-assist version."""
    from bmad_assist import __version__

    return JSONResponse({"version": __version__})


routes = [
    Route("/api/version", get_version, methods=["GET"]),
    Route("/api/status", get_status, methods=["GET"]),
    Route("/api/state", get_state, methods=["GET"]),
    Route("/api/stories", get_stories, methods=["GET"]),
    Route("/api/epics/{epic_id}", get_epic_details, methods=["GET"]),
    Route("/api/epics/{epic_id}/stories/{story_id}", get_story_in_epic, methods=["GET"]),
    Route("/api/epics/{epic_id}/metrics", get_epic_metrics, methods=["GET"]),
]
