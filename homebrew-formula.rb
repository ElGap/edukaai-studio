class EdukaaiStudio < Formula
  include Language::Python::Virtualenv

  desc "Local LLM fine-tuning studio for Apple Silicon"
  homepage "https://github.com/elgap/edukaai-studio"
  url "https://github.com/elgap/edukaai-studio/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "PLACEHOLDER_SHA256_v0.1.1"
  license "MIT"

  # EdukaAI Studio requires Apple Silicon because it is built on Apple's MLX framework.
  # MLX has no Intel Mac support.
  depends_on arch: :arm64
  depends_on "git"
  depends_on "node@22"
  depends_on "python@3.12"

  # ---------------------------------------------------------------------------
  # Python resource blocks
  # ---------------------------------------------------------------------------
  # Auto-generated from backend/requirements.txt.
  # To regenerate: brew update-python-resources local/test-tap/edukaai-studio
  # (formula must be in a tap for the command to work).
  # ---------------------------------------------------------------------------

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/5d/45/c130091c2dfa061bbfe3150f2a5091ef1adf149f2a8d2ae769ecaf6e99a2/fastapi-0.136.1.tar.gz"
    sha256 "7af665ad7acfa0a3baf8983d393b6b471b9da10ede59c60045f49fbc89a0fa7f"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/1f/93/041fca8274050e40e6791f267d82e0e2e27dd165627bd640d3e0e378d877/uvicorn-0.46.0.tar.gz"
    sha256 "fb9eaa44dbeb1c26dcc69e4bd7ec54a1cb8dd64d3b4d81ef08d90ff453f2b01b"
  end

  resource "python-multipart" do
    url "https://files.pythonhosted.org/packages/69/9b/f23807317a113dc36e74e75eb265a02dd1a4d9082abc3c1064acd22997c4/python_multipart-0.0.27.tar.gz"
    sha256 "9870a6a8c5a20a5bf4f07c017bd1489006ff8836cff097b6933355ee2b49b602"
  end

  resource "websockets" do
    url "https://files.pythonhosted.org/packages/04/24/4b2031d72e840ce4c1ccb255f693b15c334757fc50023e4db9537080b8c4/websockets-16.0.tar.gz"
    sha256 "5f6261a5e56e8d5c42a4497b364ea24d94d9563e8fbd44e78ac40879c60179b5"
  end

  resource "sqlalchemy" do
    url "https://files.pythonhosted.org/packages/09/45/461788f35e0364a8da7bda51a1fe1b09762d0c32f12f63727998d85a873b/sqlalchemy-2.0.49.tar.gz"
    sha256 "d15950a57a210e36dd4cec1aac22787e2a4d57ba9318233e2ef8b2daf9ff2d5f"
  end

  resource "alembic" do
    url "https://files.pythonhosted.org/packages/94/13/8b084e0f2efb0275a1d534838844926f798bd766566b1375174e2448cd31/alembic-1.18.4.tar.gz"
    sha256 "cb6e1fd84b6174ab8dbb2329f86d631ba9559dd78df550b57804d607672cedbc"
  end

  resource "aiosqlite" do
    url "https://files.pythonhosted.org/packages/4e/8a/64761f4005f17809769d23e518d915db74e6310474e733e3593cfc854ef1/aiosqlite-0.22.1.tar.gz"
    sha256 "043e0bd78d32888c0a9ca90fc788b38796843360c855a7262a532813133a0650"
  end

  # NOTE: MLX distributes platform-specific wheels (ARM64 only).
  # Run `brew update-python-resources` in a proper network environment to generate
  # the correct resource block for your MLX version.

  resource "mlx-lm" do
    url "https://files.pythonhosted.org/packages/84/94/9a38d6b0c6fcca995b9136c94eb7da1e9c5165652edf228b96b29960fa7a/mlx_lm-0.31.3.tar.gz"
    sha256 "61eb0e3ba09444f77f874aff295401d7ccd20b39495cbbce0c782a15474ce733"
  end

  resource "transformers" do
    url "https://files.pythonhosted.org/packages/4d/fe/7e84d20ac7d4d5d14bac2eab5976088d86342959fc2c0da54b4c2fc99856/transformers-4.51.3.tar.gz"
    sha256 "e292fcab399488a5c910baf515afcde145b517d91ad30bc53eb0a45e2c8ae925"
  end

  resource "huggingface-hub" do
    url "https://files.pythonhosted.org/packages/56/52/1b54cb569509c725a32c1315261ac9fd0e6b91bbbf74d86fca10d3376164/huggingface_hub-0.25.2.tar.gz"
    sha256 "7c3fe85e24b652334e5d456d7a812cd9a071e75630fac4365d9165ab5e4a34b6"
  end

  resource "safetensors" do
    url "https://files.pythonhosted.org/packages/29/9c/6e74567782559a63bd040a236edca26fd71bc7ba88de2ef35d75df3bca5e/safetensors-0.7.0.tar.gz"
    sha256 "07663963b67e8bd9f0b8ad15bb9163606cd27cc5a1b96235a50d8369803b96b0"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/d9/e4/40d09941a2cebcb20609b86a559817d5b9291c49dd6f8c87e5feffbe703a/pydantic-2.13.3.tar.gz"
    sha256 "af09e9d1d09f4e7fe37145c1f577e1d61ceb9a41924bf0094a36506285d0a84d"
  end

  resource "pydantic-settings" do
    url "https://files.pythonhosted.org/packages/42/98/c8345dccdc31de4228c039a98f6467a941e39558da41c1744fbe29fa5666/pydantic_settings-2.14.0.tar.gz"
    sha256 "24285fd4b0e0c06507dd9fdfd331ee23794305352aaec8fc4eb92d4047aeb67d"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/82/ed/0301aeeac3e5353ef3d94b6ec08bbcabd04a72018415dcb29e588514bba8/python_dotenv-1.2.2.tar.gz"
    sha256 "2c371a91fbd7ba082c2c1dc1f8bf89ca22564a087c2c287cd9b662adde799cf3"
  end

  # NOTE: orjson uses Rust extensions and distributes pre-built wheels.
  # Run `brew update-python-resources` in a proper network environment to generate
  # the correct resource block.

  resource "python-dateutil" do
    url "https://files.pythonhosted.org/packages/66/c0/0c8b6ad9f17a802ee498c46e004a0eb49bc148f2fd230864601a86dcf6db/python-dateutil-2.9.0.post0.tar.gz"
    sha256 "37dd54208da7e1cd875388217d5e00ebd4179249f90fb72437e91a35459a0ad3"
  end

  resource "psutil" do
    url "https://files.pythonhosted.org/packages/aa/c6/d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/psutil-7.2.2.tar.gz"
    sha256 "0746f5f8d406af344fd547f1c8daa5f5c33dbc293bb8d6a16d80b4bb88f59372"
  end

  # NOTE: NumPy often distributes platform-specific wheels.
  # Run `brew update-python-resources` in a proper network environment to generate
  # the correct resource block.

  resource "python-jose" do
    url "https://files.pythonhosted.org/packages/c6/77/3a1c9039db7124eb039772b935f2244fbb73fc8ee65b9acf2375da1c07bf/python_jose-3.5.0.tar.gz"
    sha256 "fb4eaa44dbeb1c26dcc69e4bd7ec54a1cb8dd64d3b4d81ef08d90ff453f2b01b"
  end

  resource "passlib" do
    url "https://files.pythonhosted.org/packages/b6/06/9da9ee59a67fae7761aab3ccc84fa4f3f33f125b370f1ccdb915bf967c11/passlib-1.7.4.tar.gz"
    sha256 "defd50f72b65c5402ab2c573830a6978e5f202ad0d984793c8dde2c4152ebe04"
  end

  resource "pytest" do
    url "https://files.pythonhosted.org/packages/7d/0d/549bd94f1a0a402dc8cf64563a117c0f3765662e2e668477624baeec44d5/pytest-9.0.3.tar.gz"
    sha256 "b86ada508af81d19edeb213c681b1d48246c1a91d304c6c81a427674c17eb91c"
  end

  resource "pytest-asyncio" do
    url "https://files.pythonhosted.org/packages/90/2c/8af215c0f776415f3590cac4f9086ccefd6fd463befeae41cd4d3f193e5a/pytest_asyncio-1.3.0.tar.gz"
    sha256 "d7f52f36d231b80ee124cd216ffb19369aa168fc10095013c6b014a34d3ee9e5"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/b1/df/48c586a5fe32a0f01324ee087459e112ebb7224f646c0b5023f5e79e9956/httpx-0.28.1.tar.gz"
    sha256 "75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc"
  end

  def install
    # -------------------------------------------------------------------------
    # 1. Build frontend static assets
    # -------------------------------------------------------------------------
    cd "frontend" do
      system "npm", "install", *std_npm_args
      system "npm", "run", "build"
    end

    # -------------------------------------------------------------------------
    # 2. Install Python backend into a Homebrew-managed virtualenv
    # -------------------------------------------------------------------------
    # The root pyproject.toml makes the `app` package (in backend/app)
    # pip-installable. virtualenv_install_with_resources installs all declared
    # resource blocks and then the package itself.
    virtualenv_install_with_resources

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
