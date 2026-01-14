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
     pkgs.gcc
     pkgs.gcc.cc.lib
     pkgs.glibc
     pkgs.stdenv.cc.cc.lib
     pkgs.libstdcxx5
   ];
   env = {
     # Put local wheels first, fall back to the Nix store last
     PYTHONPATH = ".pythonlibs/lib/python3.11/site-packages:${pkgs.python311}/lib/python3.11/site-packages";
     # Ensure libstdc++ is discoverable for compiled wheels (e.g., pandas)
     LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.gcc.cc.lib}/lib:${pkgs.libstdcxx5}/lib";

   };
 }
