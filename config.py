import os
import toml


class Config:
    _instance = None  # Singleton instance

    def __new__(cls, config_file_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_file_path is None:
                config_file_path = os.path.join('./config.toml')
            cls._instance.config = toml.load(config_file_path)  # Load the config only once

            # Dynamically set attributes based on TOML config
            for section, values in cls._instance.config.items():
                for key, value in values.items():
                    setattr(cls._instance, key, value)
            cls._instance._apply_env_overrides()

        return cls._instance

    def _apply_env_overrides(self):
        override_map = {
            "dataset": "RAGTEST_DATASET",
            "source_dir": "RAGTEST_SOURCE_DIR",
            "persist_dir": "RAGTEST_PERSIST_DIR",
            "split_type": "RAGTEST_SPLIT_TYPE",
            "chunk_size": "RAGTEST_CHUNK_SIZE",
            "n": "RAGTEST_N",
        }
        for attr, env_key in override_map.items():
            if env_key not in os.environ:
                continue
            raw_value = os.environ[env_key]
            current = getattr(self, attr, None)
            if isinstance(current, bool):
                value = raw_value.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                value = int(raw_value)
            elif isinstance(current, float):
                value = float(raw_value)
            else:
                value = raw_value
            setattr(self, attr, value)


class GlobalVar:
    query_number = 0

    @staticmethod
    def set_query_number(num):
        GlobalVar.query_number = num

    @staticmethod
    def get_query_number():
        return GlobalVar.query_number


if __name__ == '__main__':
    # Usage: Create a global instance
    cfg = Config()

    # Now you can access config values directly as attributes:
    print(cfg.test_init_total_number_documents)  # Outputs: 20
    print(cfg.api_key)  # Outputs the API key from the toml file
