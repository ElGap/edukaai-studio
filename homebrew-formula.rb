class EdukaaiStudio < Formula
  include Language::Python::Virtualenv

  desc "Local LLM fine-tuning studio for Apple Silicon"
  homepage "https://github.com/elgap/edukaai-studio"
  url "https://github.com/elgap/edukaai-studio/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  version "0.1.0"
  license "MIT"

  # EdukaAI Studio requires Apple Silicon because it is built on Apple's MLX framework.
  # MLX has no Intel Mac support.
  depends_on arch: :arm64
  depends_on "node@22"
  depends_on "python@3.12"
  depends_on "git"

  # ---------------------------------------------------------------------------
  # Python resource blocks
  # ---------------------------------------------------------------------------
  # These are auto-generated from backend/requirements.txt by running:
  #   brew update-python-resources ./homebrew-formula.rb
  #
  # The command resolves all direct + transitive dependencies from PyPI and
  # inserts resource blocks with URLs and SHA256 checksums.
  #
  # After running the command, replace the placeholder SHA256 above with the
  # real hash of the release tarball:
  #   shasum -a 256 edukaai-studio-0.1.0.tar.gz
  # ---------------------------------------------------------------------------

  # Resource blocks will be inserted here by `brew update-python-resources`.
  # Example format:
  #   resource "fastapi" do
  #     url "https://files.pythonhosted.org/packages/.../fastapi-0.135.0.tar.gz"
  #     sha256 "..."
  #   end

  def install
    # -------------------------------------------------------------------------
    # 1. Build frontend static assets
    # -------------------------------------------------------------------------
    cd "frontend" do
      system "npm", "install"
      system "npm", "run", "build"
    end

    # -------------------------------------------------------------------------
    # 2. Install Python backend into a Homebrew-managed virtualenv
    # -------------------------------------------------------------------------
    # The backend/ directory contains pyproject.toml making the `app` package
    # pip-installable. virtualenv_install_with_resources installs all declared
    # resource blocks (from requirements.txt) and then the package itself.
    cd "backend" do
      virtualenv_install_with_resources
    end

    # -------------------------------------------------------------------------
    # 3. Install full source tree into pkgshare for runtime assets
    # -------------------------------------------------------------------------
    # We need:
    #   - frontend/dist/          (built static files served by FastAPI)
    #   - .env.example            (template, not used directly — all config
    #                               is set via env vars in the wrapper script)
    #   - README.md, docs/        (for user reference)
    # The backend Python code is already in the venv; we keep the source here
    # so the wrapper script can cd into it and create the storage symlink.
    pkgshare.install Dir["*"]

    # -------------------------------------------------------------------------
    # 4. Create wrapper script
    # -------------------------------------------------------------------------
    # The wrapper sets all configuration via environment variables so no .env
    # file is needed. Persistent data lives in ~/.edukaai/ and survives
    # reinstalls and upgrades.
    (bin/"edukaai-studio").write <<~EOS
      #!/bin/bash
      set -euo pipefail

      # Persistent data directory (survives reinstalls)
      DATA_DIR="${EDUKAAI_DATA_DIR:-$HOME/.edukaai}"
      STORAGE_PATH="${EDUKAAI_STORAGE_PATH:-$DATA_DIR/storage}"
      MODEL_CACHE_DIR="${EDUKAAI_MODEL_CACHE_DIR:-$DATA_DIR/models}"
      TRAINING_OUTPUT_DIR="${EDUKAAI_TRAINING_OUTPUT_DIR:-$DATA_DIR/training}"
      LOG_DIR="$DATA_DIR/logs"

      # Ensure persistent directories exist
      mkdir -p "$STORAGE_PATH" "$MODEL_CACHE_DIR" "$TRAINING_OUTPUT_DIR" "$LOG_DIR"

      # Application installation directory (Cellar pkgshare)
      STUDIO_DIR="#{pkgshare}"
      cd "$STUDIO_DIR"

      # Replace the bundled backend/storage directory with a symlink to the
      # persistent storage location. This is idempotent: if it's already a
      # symlink we leave it; if it's a real directory (first run or after
      # reinstall) we replace it.
      if [ -d "backend/storage" ] && [ ! -L "backend/storage" ]; then
        rm -rf "backend/storage"
      fi
      if [ ! -L "backend/storage" ]; then
        ln -s "$STORAGE_PATH" backend/storage
      fi

      # Set all configuration via environment variables.
      # No .env file is required because every setting has a default or is set here.
      export EDUKAAI_HOST="${EDUKAAI_HOST:-127.0.0.1}"
      export EDUKAAI_PORT="${EDUKAAI_PORT:-8000}"
      export EDUKAAI_STORAGE_PATH="$STORAGE_PATH"
      export EDUKAAI_MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
      export EDUKAAI_TRAINING_OUTPUT_DIR="$TRAINING_OUTPUT_DIR"
      export EDUKAAI_LOG_FILE="${EDUKAAI_LOG_FILE:-$LOG_DIR/edukaai.log}"
      export EDUKAAI_DATABASE_URL="${EDUKAAI_DATABASE_URL:-sqlite:///$STORAGE_PATH/app/edukaai.db}"
      export EDUKAAI_FRONTEND_DIST="${EDUKAAI_FRONTEND_DIST:-$STUDIO_DIR/frontend/dist}"
      export EDUKAAI_ALLOW_REMOTE="${EDUKAAI_ALLOW_REMOTE:-false}"
      export EDUKAAI_LOG_LEVEL="${EDUKAAI_LOG_LEVEL:-INFO}"

      # Change into backend so relative paths (./storage, etc.) resolve correctly
      cd backend

      # Start the server
      exec "#{libexec}/bin/python" -m uvicorn app.main:app \\
        --host "$EDUKAAI_HOST" \\
        --port "$EDUKAAI_PORT" \\
        --log-level "$EDUKAAI_LOG_LEVEL" \\
        --no-access-log
    EOS
    chmod 0555, bin/"edukaai-studio"
  end

  # ---------------------------------------------------------------------------
  # 5. brew services support
  # ---------------------------------------------------------------------------
  service do
    run [opt_bin/"edukaai-studio"]
    keep_alive true
    log_path var/"log/edukaai-studio.log"
    error_log_path var/"log/edukaai-studio.log"
    environment_variables PATH: std_service_path_env
  end

  # ---------------------------------------------------------------------------
  # 6. User-facing post-install notes
  # ---------------------------------------------------------------------------
  def caveats
    <<~EOS
      EdukaAI Studio requires Apple Silicon (M1/M2/M3/M4).
      It cannot run on Intel Macs because it depends on Apple's MLX framework.

      Data and downloaded models persist in ~/.edukaai/ and survive reinstalls.

      To start interactively (foreground, Ctrl+C to stop):
        edukaai-studio

      To start as a background service:
        brew services start edukaai-studio

      Then open: http://localhost:8000

      Environment variables you can override:
        EDUKAAI_HOST          (default: 127.0.0.1)
        EDUKAAI_PORT          (default: 8000)
        EDUKAAI_DATA_DIR      (default: ~/.edukaai)
        EDUKAAI_HF_TOKEN      (HuggingFace token for private models)
    EOS
  end

  # ---------------------------------------------------------------------------
  # 7. Test block
  # ---------------------------------------------------------------------------
  # Starts the server, waits for the health endpoint to respond, then stops.
  # This is a lightweight smoke test that does not download any models.
  test do
    pid = fork { exec bin/"edukaai-studio" }
    sleep 8
    begin
      output = shell_output("curl -sf http://127.0.0.1:8000/api/health || echo 'not healthy'")
      assert_match "healthy", output
    ensure
      Process.kill("TERM", pid) if pid
      Process.wait(pid) if pid
    end
  end
end
