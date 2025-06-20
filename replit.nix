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
    pkgs.python311
    pkgs.nodePackages.pyright
    pkgs.black
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
  ];
  env = {
    PYTHONPATH = "${pkgs.python311}/lib/python3.11/site-packages";
  };
}