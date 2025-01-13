{ pkgs }: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.libxcrypt
    pkgs.glibcLocales
    pkgs.python311Packages.hypercorn
    # Core Python and development tools
    pkgs.python3
    pkgs.poetry
    pkgs.git

    # Build dependencies
    pkgs.gcc
    pkgs.pkg-config

    # System libraries
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
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
    POETRY_VERSION = "1.7.1";
    POETRY_HOME = "${pkgs.poetry}";
    POETRY_VIRTUALENVS_CREATE = "true";
    POETRY_VIRTUALENVS_IN_PROJECT = "true";
    POETRY_CACHE_DIR = "/tmp/.cache/pypoetry";
    PYTHONUNBUFFERED = "1";
    FLASK_APP = "api.app";
    FLASK_ENV = "development";
  };
}