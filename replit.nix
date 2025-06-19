{ pkgs }: {
  # Force cache invalidation - Python 3.11 environment rebuild
  # Updated: 2025-06-19 to resolve libpython3.10.so.1.0 shared library issue
  deps = [
    pkgs.python311
    pkgs.postgresql_16
    pkgs.arrow-cpp
    pkgs.cargo
    pkgs.glibcLocales
    pkgs.libiconv
    pkgs.libxcrypt
    pkgs.pgadmin4
    pkgs.pkg-config
    pkgs.rustc
  ];
  env = {
    LANG = "en_US.UTF-8";
    # Ensure all scripts use the Python from Nix
    PYTHONPATH = "${pkgs.python311}/lib/python3.11/site-packages";
    # Set the library path for compiled modules
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.python311
      pkgs.libiconv
      pkgs.libxcrypt
    ];
  };
}