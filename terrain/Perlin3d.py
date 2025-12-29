import torch


def perlin_noise_3d_torch(
    size, scale=32.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=0, device="cuda"
):
    """
    Generate 3D Perlin noise as a torch.Tensor of shape (D, H, W).
    - size: (D, H, W) tuple
    - scale: base frequency denominator; larger => smoother
    - octaves: number of layers to sum
    - persistence: amplitude multiplier per octave
    - lacunarity: frequency multiplier per octave
    - seed: random seed for permutation table
    """
    D, H, W = size
    torch.manual_seed(seed)
    device = torch.device(device)

    # Permutation table (repeat to avoid modulus)
    p = torch.randperm(256, device=device)
    perm = torch.cat([p, p])

    # Gradient directions for 3D (12 canonical directions)
    grads = torch.tensor(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1],
        ],
        dtype=torch.float32,
        device=device,
    )

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, t):
        return a + t * (b - a)

    def noise_single(freq):
        # Grid coordinates
        z = torch.linspace(0, D, D, device=device) / scale * freq
        y = torch.linspace(0, H, H, device=device) / scale * freq
        x = torch.linspace(0, W, W, device=device) / scale * freq

        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

        # Unit cube
        X0 = torch.floor(X).to(torch.int32) & 255
        Y0 = torch.floor(Y).to(torch.int32) & 255
        Z0 = torch.floor(Z).to(torch.int32) & 255

        # Local position in cube
        xf = X - torch.floor(X)
        yf = Y - torch.floor(Y)
        zf = Z - torch.floor(Z)

        u = fade(xf)
        v = fade(yf)
        w = fade(zf)

        # Hash function to pick gradient indices
        def hash(ix, iy, iz):
            return perm[perm[perm[ix] + iy] + iz] % 12

        # Corner hashes
        g000 = grads[hash(X0, Y0, Z0)]
        g001 = grads[hash(X0, Y0, (Z0 + 1) & 255)]
        g010 = grads[hash(X0, (Y0 + 1) & 255, Z0)]
        g011 = grads[hash(X0, (Y0 + 1) & 255, (Z0 + 1) & 255)]
        g100 = grads[hash((X0 + 1) & 255, Y0, Z0)]
        g101 = grads[hash((X0 + 1) & 255, Y0, (Z0 + 1) & 255)]
        g110 = grads[hash((X0 + 1) & 255, (Y0 + 1) & 255, Z0)]
        g111 = grads[hash((X0 + 1) & 255, (Y0 + 1) & 255, (Z0 + 1) & 255)]

        # Vectors from corners to point
        x0, y0, z0 = xf, yf, zf
        x1, y1, z1 = xf - 1, yf - 1, zf - 1

        # Dot products
        n000 = g000[..., 0] * x0 + g000[..., 1] * y0 + g000[..., 2] * z0
        n001 = g001[..., 0] * x0 + g001[..., 1] * y0 + g001[..., 2] * z1
        n010 = g010[..., 0] * x0 + g010[..., 1] * y1 + g010[..., 2] * z0
        n011 = g011[..., 0] * x0 + g011[..., 1] * y1 + g011[..., 2] * z1
        n100 = g100[..., 0] * x1 + g100[..., 1] * y0 + g100[..., 2] * z0
        n101 = g101[..., 0] * x1 + g101[..., 1] * y0 + g101[..., 2] * z1
        n110 = g110[..., 0] * x1 + g110[..., 1] * y1 + g110[..., 2] * z0
        n111 = g111[..., 0] * x1 + g111[..., 1] * y1 + g111[..., 2] * z1

        # Trilinear interpolation with fade
        nx00 = lerp(n000, n100, u)
        nx01 = lerp(n001, n101, u)
        nx10 = lerp(n010, n110, u)
        nx11 = lerp(n011, n111, u)

        nxy0 = lerp(nx00, nx10, v)
        nxy1 = lerp(nx01, nx11, v)

        nxyz = lerp(nxy0, nxy1, w)

        return nxyz  # roughly in [-1, 1]

    noise = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        noise += amplitude * noise_single(frequency)
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    noise /= max_amp  # normalize to [-1, 1] range-ish
    return noise


# Example
if __name__ == "__main__":
    vol = perlin_noise_3d_torch(
        size=(64, 64, 64),
        scale=32.0,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        seed=42,
        device="cuda",
    )
    print(vol)
