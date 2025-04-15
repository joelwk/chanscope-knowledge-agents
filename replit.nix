{ pkgs }: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.pgadmin4
    pkgs.pkg-config
    pkgs.arrow-cpp
    pkgs.glibcLocales
    pkgs.libxcrypt
    pkgs.python310
    pkgs.poetry
    pkgs.nodePackages.pyright
    pkgs.black
    pkgs.python310Packages.pip
    pkgs.python310Packages.poetry-core
    pkgs.python310Packages.virtualenv
  ];
  env = {
    POETRY_VIRTUALENVS_IN_PROJECT = "true";
    POETRY_VIRTUALENVS_CREATE = "true";
    PYTHONPATH = "${pkgs.python310}/lib/python3.10/site-packages";
  };
}