{
  description = "Flake with TensorFlow";

  # From cmpbayes, 2022-02-21.
  inputs.nixpkgs.url =
    "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";
  inputs.cmpbayes.url = "github:dpaetzel/cmpbayes";
  inputs.cmpbayes.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, cmpbayes }:
    let system = "x86_64-linux";
    in with import nixpkgs {
      inherit system;
    };

    let python = python39;
    in rec {

      devShell.${system} = pkgs.mkShell {

        packages = with python.pkgs; [
          cmpbayes.defaultPackage."${system}"
          ipython
          python

          matplotlib
          numpy
          pandas

          tensorflow

          python.pkgs.opencv3
          tqdm
          scikit-learn
          keras
        ];
      };
    };
}
