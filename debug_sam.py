import sys

print("🔍 Debugging Imports...")
print(f"Python Executable: {sys.executable}")

try:
    import huggingface_hub
    print(f"✅ huggingface_hub version: {huggingface_hub.__version__}")
except ImportError as e:
    print(f"❌ Failed to import huggingface_hub: {e}")

try:
    import transformers
    print(f"✅ transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Failed to import transformers: {e}")

print("\nAttempting specific imports...")

try:
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    print("✅ GatedRepoError found.")
except ImportError as e:
    print(f"❌ Failed to import GatedRepoError: {e}")

try:
    from transformers import Sam3Processor
    print("✅ Sam3Processor found.")
except ImportError as e:
    print(f"❌ Failed to import Sam3Processor: {e}")
    print("   -> This likely means the 'transformers' library needs an update or installation from source.")

try:
    from transformers import Sam3Model
    print("✅ Sam3Model found.")
except ImportError as e:
    print(f"❌ Failed to import Sam3Model: {e}")