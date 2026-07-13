class AudioCleaningPipeline:
    def __init__(self, config=None):
        self.config = config
        self.steps = []

    def add_step(self, name, fn, enabled = True, params = None):
        if params is None:
            params = {}
        self.steps.append({
            "name": name,
            "fn": fn,
            "enabled": enabled,
            "params": params
        })

    def run(self, signal, sample_rate):
        diagnostics = {}

        for step in self.steps:
            if not step["enabled"]:
                continue

            signal = step["fn"](
                signal,
                sample_rate,
                **step["params"]
            )

            diagnostics[step["name"]] = {
                "completed": True
            }
            
        return signal, diagnostics