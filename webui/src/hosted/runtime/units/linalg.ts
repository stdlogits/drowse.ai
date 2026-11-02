// Small host-side linear algebra for the units ports (fp64): eigenvalues of a real square
// matrix (the head copy score's `eigvals(Bᵀ A)`, `[hd, hd]`), a dense solve (the pursuit's tiny
// least squares) and row-major products.

/** `a [m, k] @ b [k, n]`. */
export function matmul(a: ArrayLike<number>, b: ArrayLike<number>, m: number, k: number, n: number): Float64Array {
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i += 1) {
    for (let t = 0; t < k; t += 1) {
      const x = a[i * k + t];
      if (x === 0) continue;
      for (let j = 0; j < n; j += 1) out[i * n + j] += x * b[t * n + j];
    }
  }
  return out;
}

export function transpose(a: ArrayLike<number>, m: number, n: number): Float64Array {
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i += 1) for (let j = 0; j < n; j += 1) out[j * m + i] = a[i * n + j];
  return out;
}

/** Solve `A x = b` (`A [n, n]`) by Gaussian elimination with partial pivoting. */
export function solve(a: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array {
  const m = Float64Array.from(a);
  const x = Float64Array.from(b);
  for (let c = 0; c < n; c += 1) {
    let pivot = c;
    for (let r = c + 1; r < n; r += 1) if (Math.abs(m[r * n + c]) > Math.abs(m[pivot * n + c])) pivot = r;
    if (m[pivot * n + c] === 0) throw new RangeError("singular system");
    if (pivot !== c) {
      for (let j = 0; j < n; j += 1) [m[c * n + j], m[pivot * n + j]] = [m[pivot * n + j], m[c * n + j]];
      [x[c], x[pivot]] = [x[pivot], x[c]];
    }
    for (let r = c + 1; r < n; r += 1) {
      const f = m[r * n + c] / m[c * n + c];
      if (f === 0) continue;
      for (let j = c; j < n; j += 1) m[r * n + j] -= f * m[c * n + j];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r -= 1) {
    let s = x[r];
    for (let j = r + 1; j < n; j += 1) s -= m[r * n + j] * x[j];
    x[r] = s / m[r * n + r];
  }
  return x;
}

/** Eigenvalues of a real `[n, n]` matrix: Hessenberg reduction by stabilized elimination, then
 *  the shifted QR iteration (EISPACK `elmhes` + `hqr`). Returns real and imaginary parts. */
export function eigenvalues(input: ArrayLike<number>, n: number): { re: Float64Array; im: Float64Array } {
  const a: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => input[i * n + j]));
  // elmhes
  for (let m = 1; m < n - 1; m += 1) {
    let x = 0;
    let i = m;
    for (let j = m; j < n; j += 1) {
      if (Math.abs(a[j][m - 1]) > Math.abs(x)) {
        x = a[j][m - 1];
        i = j;
      }
    }
    if (i !== m) {
      for (let j = m - 1; j < n; j += 1) [a[i][j], a[m][j]] = [a[m][j], a[i][j]];
      for (let j = 0; j < n; j += 1) [a[j][i], a[j][m]] = [a[j][m], a[j][i]];
    }
    if (x !== 0) {
      for (i = m + 1; i < n; i += 1) {
        let y = a[i][m - 1];
        if (y !== 0) {
          y /= x;
          a[i][m - 1] = y;
          for (let j = m; j < n; j += 1) a[i][j] -= y * a[m][j];
          for (let j = 0; j < n; j += 1) a[j][m] += y * a[j][i];
        }
      }
    }
  }
  for (let i = 2; i < n; i += 1) for (let j = 0; j < i - 1; j += 1) a[i][j] = 0;
  // hqr
  const wr = new Float64Array(n);
  const wi = new Float64Array(n);
  let anorm = 0;
  for (let i = 0; i < n; i += 1) for (let j = Math.max(i - 1, 0); j < n; j += 1) anorm += Math.abs(a[i][j]);
  let nn = n - 1;
  let t = 0;
  let p = 0, q = 0, r = 0, s = 0, w = 0, x = 0, y = 0, z = 0;
  while (nn >= 0) {
    let its = 0;
    let l: number;
    do {
      for (l = nn; l >= 1; l -= 1) {
        s = Math.abs(a[l - 1][l - 1]) + Math.abs(a[l][l]);
        if (s === 0) s = anorm;
        if (Math.abs(a[l][l - 1]) + s === s) {
          a[l][l - 1] = 0;
          break;
        }
      }
      x = a[nn][nn];
      if (l === nn) {
        wr[nn] = x + t;
        wi[nn] = 0;
        nn -= 1;
      } else {
        y = a[nn - 1][nn - 1];
        w = a[nn][nn - 1] * a[nn - 1][nn];
        if (l === nn - 1) {
          p = 0.5 * (y - x);
          q = p * p + w;
          z = Math.sqrt(Math.abs(q));
          x += t;
          if (q >= 0) {
            z = p + (p >= 0 ? Math.abs(z) : -Math.abs(z));
            wr[nn - 1] = wr[nn] = x + z;
            if (z) wr[nn] = x - w / z;
            wi[nn - 1] = wi[nn] = 0;
          } else {
            wr[nn - 1] = wr[nn] = x + p;
            wi[nn - 1] = -(wi[nn] = z);
          }
          nn -= 2;
        } else {
          if (its === 60) throw new Error("eigenvalues: too many iterations");
          if (its === 10 || its === 20) {
            t += x;
            for (let i = 0; i <= nn; i += 1) a[i][i] -= x;
            s = Math.abs(a[nn][nn - 1]) + Math.abs(a[nn - 1][nn - 2]);
            y = x = 0.75 * s;
            w = -0.4375 * s * s;
          }
          its += 1;
          let m: number;
          for (m = nn - 2; m >= l; m -= 1) {
            z = a[m][m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
            q = a[m + 1][m + 1] - z - r - s;
            r = a[m + 2][m + 1];
            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m === l) break;
            const u = Math.abs(a[m][m - 1]) * (Math.abs(q) + Math.abs(r));
            const v = Math.abs(p) * (Math.abs(a[m - 1][m - 1]) + Math.abs(z) + Math.abs(a[m + 1][m + 1]));
            if (u + v === v) break;
          }
          for (let i = m + 2; i <= nn; i += 1) {
            a[i][i - 2] = 0;
            if (i !== m + 2) a[i][i - 3] = 0;
          }
          for (let k = m; k <= nn - 1; k += 1) {
            if (k !== m) {
              p = a[k][k - 1];
              q = a[k + 1][k - 1];
              r = 0;
              if (k !== nn - 1) r = a[k + 2][k - 1];
              if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            const root = Math.sqrt(p * p + q * q + r * r);
            if ((s = p >= 0 ? root : -root) !== 0) {
              if (k === m) {
                if (l !== m) a[k][k - 1] = -a[k][k - 1];
              } else {
                a[k][k - 1] = -s * x;
              }
              p += s;
              x = p / s;
              y = q / s;
              z = r / s;
              q /= p;
              r /= p;
              for (let j = k; j <= nn; j += 1) {
                p = a[k][j] + q * a[k + 1][j];
                if (k !== nn - 1) {
                  p += r * a[k + 2][j];
                  a[k + 2][j] -= p * z;
                }
                a[k + 1][j] -= p * y;
                a[k][j] -= p * x;
              }
              const mmin = nn < k + 3 ? nn : k + 3;
              for (let i = l; i <= mmin; i += 1) {
                p = x * a[i][k] + y * a[i][k + 1];
                if (k !== nn - 1) {
                  p += z * a[i][k + 2];
                  a[i][k + 2] -= p * r;
                }
                a[i][k + 1] -= p * q;
                a[i][k] -= p;
              }
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return { re: wr, im: wi };
}

/** Elhage et al.'s copy score `Σλ / Σ|λ|` of `M` (`[n, n]`; 0 when every eigenvalue is 0). */
export function copyScore(m: ArrayLike<number>, n: number): number {
  const { re, im } = eigenvalues(m, n);
  let real = 0;
  let abs = 0;
  for (let i = 0; i < n; i += 1) {
    real += re[i];
    abs += Math.hypot(re[i], im[i]);
  }
  return abs > 0 ? real / abs : 0;
}
