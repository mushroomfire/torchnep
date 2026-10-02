// NEP_CPU reference driver: E / F / per-atom virial for every frame of a plain
// text file, in double precision.
//   input : nframes, then per frame: N / "ax ay az bx by bz cx cy cz" / N x "type x y z"
//   output: per frame N lines "e fx fy fz vxx vxy vxz vyx vyy vyz vzx vzy vzz"
#include "nep.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "usage: driver nep.txt frames.txt out.txt\n";
    return 1;
  }
  NEP nep(argv[1]);
  std::ifstream in(argv[2]);
  FILE* out = fopen(argv[3], "w");
  int nframes;
  in >> nframes;
  for (int f = 0; f < nframes; ++f) {
    int N;
    in >> N;
    double a[9];
    for (int k = 0; k < 9; ++k) in >> a[k];
    // NEP_CPU: box = ax, bx, cx, ay, by, cy, az, bz, cz (lattice vectors as columns)
    std::vector<double> box = {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
    std::vector<int> type(N);
    std::vector<double> pos(3 * N), pe(N), force(3 * N), virial(9 * N);
    for (int n = 0; n < N; ++n) in >> type[n] >> pos[n] >> pos[n + N] >> pos[n + 2 * N];
    nep.compute(type, box, pos, pe, force, virial);
    for (int n = 0; n < N; ++n) {
      fprintf(out, "%.17g %.17g %.17g %.17g", pe[n], force[n], force[n + N], force[n + 2 * N]);
      for (int k = 0; k < 9; ++k) fprintf(out, " %.17g", virial[n + k * N]);
      fprintf(out, "\n");
    }
  }
  fclose(out);
  return 0;
}
