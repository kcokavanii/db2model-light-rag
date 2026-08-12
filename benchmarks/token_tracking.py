from typing import Any, Dict


class TokenTrackingClient:
    def __init__(self, client: Any):
        self.client = client
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def create(self, *args, **kwargs) -> Any:
        resp = await self.client.create(*args, **kwargs)

        usage = getattr(resp, "usage", None)
        if usage:
            # Handle both dict-like and object-like usage
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens")
            else:
                prompt = getattr(usage, "prompt_tokens", 0)
                completion = getattr(usage, "completion_tokens", 0)
                total = getattr(usage, "total_tokens", None)

            if total is None:
                total = prompt + completion

            self.total_usage["prompt_tokens"] += prompt
            self.total_usage["completion_tokens"] += completion
            self.total_usage["total_tokens"] += total

        return resp

    def get_usage(self) -> Dict[str, int]:
        return dict(self.total_usage)

    def reset_usage(self) -> None:
        for k in self.total_usage:
            self.total_usage[k] = 0
