import wave
import struct
import heapq

# ------------------------
# Étape 1 : lecture audio
# ------------------------
def read_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
        samples = struct.unpack('<' + 'h'*params.nframes*params.nchannels, frames)
    return list(samples), params

# ------------------------
# Étape 2 : segmentation adaptative
# ------------------------
def segment_frames(samples, frame_size=1024, threshold=500):
    frames = []
    i = 0
    while i < len(samples):
        window = samples[i:i+frame_size]
        if max(window) - min(window) < threshold:
            # silence/drones → bloc long
            end = i + frame_size*4
        else:
            end = i + frame_size
        frames.append(samples[i:end])
        i = end
    return frames

# ------------------------
# Étape 3 : prédiction linéaire simple
# ------------------------
def lpc_predict(frame, order=2):
    preds = []
    for i in range(len(frame)):
        pred = sum(frame[i-j-1] for j in range(min(i, order))) // order if i>=order else 0
        preds.append(pred)
    # résidu
    residual = [frame[i] - preds[i] for i in range(len(frame))]
    return residual

# ------------------------
# Étape 4 : codage entropique léger (Huffman adaptatif simplifié)
# ------------------------
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(residual):
    freq_table = {}
    for r in residual:
        freq_table[r] = freq_table.get(r,0)+1
    heap = [HuffmanNode(s,f) for s,f in freq_table.items()]
    heapq.heapify(heap)
    while len(heap)>1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = HuffmanNode(freq=n1.freq+n2.freq)
        parent.left, parent.right = n1, n2
        heapq.heappush(heap,parent)
    return heap[0]

def build_codes(node, prefix='', codebook={}):
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        build_codes(node.left, prefix+'0', codebook)
        build_codes(node.right, prefix+'1', codebook)
    return codebook

def huffman_encode(residual):
    tree = build_huffman_tree(residual)
    codebook = build_codes(tree)
    encoded = ''.join(codebook[r] for r in residual)
    return encoded, codebook

def huffman_decode(encoded, codebook):
    inverse = {v:k for k,v in codebook.items()}
    decoded = []
    buffer = ''
    for bit in encoded:
        buffer += bit
        if buffer in inverse:
            decoded.append(inverse[buffer])
            buffer = ''
    return decoded

# ------------------------
# Étape 5 : compression complète
# ------------------------
def compress_audio(samples):
    frames = segment_frames(samples)
    compressed_frames = []
    codebooks = []
    for frame in frames:
        residual = lpc_predict(frame)
        encoded, codebook = huffman_encode(residual)
        compressed_frames.append(encoded)
        codebooks.append(codebook)
    return compressed_frames, codebooks

# ------------------------
# Étape 6 : décompression complète
# ------------------------
def decompress_audio(compressed_frames, codebooks, order=2):
    reconstructed = []
    for encoded, codebook in zip(compressed_frames, codebooks):
        residual = huffman_decode(encoded, codebook)
        frame = []
        for i in range(len(residual)):
            pred = sum(frame[i-j-1] for j in range(min(i, order))) // order if i>=order else 0
            frame.append(residual[i]+pred)
        reconstructed.extend(frame)
    return reconstructed

# ------------------------
# Exemple d'utilisation
# ------------------------
if __name__ == '__main__':
    samples, params = read_wav('input.wav')
    compressed, codebooks = compress_audio(samples)
    reconstructed = decompress_audio(compressed, codebooks)

    # Écriture d'un fichier WAV reconstruit
    with wave.open('output.wav','wb') as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(struct.pack('<'+'h'*len(reconstructed), *reconstructed))
