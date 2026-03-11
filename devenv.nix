{
  pkgs,
  lib,
  config,
  ...
}:
{
  dotenv.enable = true;
  packages = [ pkgs.ffmpeg ];
  languages.python = {
    enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };
}
