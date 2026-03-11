{
  pkgs,
  lib,
  config,
  ...
}:
{
  dotenv.enable = true;
  languages.python = {
    enable = true;
    uv.enable = true;
    uv.sync.enable = true;
  };
}
