class EdukaaiStudio < Formula
  desc "Local LLM fine-tuning studio for Apple Silicon"
  homepage "https://github.com/elgap/edukaai-studio"
  version "0.1.1"
  license "MIT"

  # Apple Silicon only — MLX framework has no Intel Mac support
  depends_on arch: :arm64

  on_macos do
    on_arm do
      url "https://github.com/elgap/edukaai-studio/releases/download/v#{version}/edukaai-studio-#{version}-darwin-arm64.tar.gz"
      sha256 "5dbcea266cb0392bcabd22f5e00d366af4974b5e8d7569389b6af7a627f08fdc"
    end
  end

  def install
    # The release tarball contains a self-contained bundle:
    #   edukaai-studio/
    #     edukaai-studio      # launcher script
    #     .venv/              # Python virtualenv with all deps
    #     dist/               # built frontend assets
    #     app/                # backend source
    #     run.py              # alternative launcher
    #     .env.example
    #     README.md
    prefix.install Dir["*"]
    bin.install_symlink prefix/"edukaai-studio"
  end

  service do
    run [opt_bin/"edukaai-studio"]
    keep_alive true
    log_path var/"log/edukaai-studio.log"
    error_log_path var/"log/edukaai-studio.log"
    environment_variables PATH: std_service_path_env
  end

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

  test do
    # Start server, curl health endpoint, then stop
    pid = fork { exec bin/"edukaai-studio" }
    sleep 10
    begin
      output = shell_output("curl -sf http://127.0.0.1:8000/api/health || echo 'not healthy'")
      assert_match "healthy", output
    ensure
      Process.kill("TERM", pid) if pid
      Process.wait(pid) if pid
    end
  end
end
