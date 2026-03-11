{
  pkgs,
  lib,
  config,
  ...
}:
{
  dotenv.enable = true;
  packages = [ pkgs.ffmpeg pkgs.zlib ];
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];
  languages.python = {
    enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };
}
