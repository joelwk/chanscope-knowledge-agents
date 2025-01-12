{pkgs}: {
  deps = [
    # Core Python and development tools
    pkgs.python311
    pkgs.poetry
    pkgs.git

    # Build dependencies
    pkgs.gcc
    pkgs.pkg-config
    
    # System libraries that might be needed
    pkgs.openssl
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      "${pkgs.zlib}/lib"
      "${pkgs.openssl}/lib"
    ];
    PYTHONBIN = "${pkgs.python311}/bin/python3.11";
    LANG = "en_US.UTF-8";
    POETRY_VIRTUALENVS_CREATE = "false";
    PYTHONUNBUFFERED = "1";
  };
} 