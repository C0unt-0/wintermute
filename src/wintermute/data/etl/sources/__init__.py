"""Auto-import all source plugins for self-registration."""

from wintermute.data.etl.sources.asm_directory import AsmDirectorySource  # noqa: F401
from wintermute.data.etl.sources.malshare import MalShareSource  # noqa: F401
from wintermute.data.etl.sources.malware_bazaar import MalwareBazaarSource  # noqa: F401
from wintermute.data.etl.sources.ms_dataset import MSDatasetSource  # noqa: F401
from wintermute.data.etl.sources.pe_files import PEFilesSource  # noqa: F401
from wintermute.data.etl.sources.synthetic import SyntheticSource  # noqa: F401
from wintermute.data.etl.sources.urlhaus import URLhausSource  # noqa: F401
