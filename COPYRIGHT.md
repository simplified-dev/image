# Copyright Notice

## Project License

Copyright 2026 CraftedFury

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

> <http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Third-Party Notices

The WebP codecs port code from, and copy tables out of, the works below. Each keeps its own copyright and licence, reproduced here as its terms require. The jar carries this file under `META-INF`, beside `LICENSE.md`, so the notices travel with a binary as well as with the source.

### libwebp - Google Inc.

<https://chromium.googlesource.com/webm/libwebp>

Classes in `dev.simplified.image.codec.webp.vp8`:

- `VP8Tables` - the keyframe token tree and the coefficient and mode probabilities (`src/dec/tree_dec.c`)
- `VP8Costs` - the bit-cost tables and residual-cost estimation (`src/enc/cost_enc.c`, `src/enc/cost_enc.h`)
- `QuantMatrix`, `TrellisQuantizer` - the quantizer bias and sharpening matrices and `TrellisQuantizeBlock` (`src/enc/quant_enc.c`)
- `DCT` - the forward and inverse DCT and Walsh-Hadamard transforms and the texture distortion (`src/dsp/enc.c`, `src/dsp/dec.c`)
- `IntraPrediction` - the intra predictors and the samples a decoder seeds along the frame's edges (`src/dsp/dec.c`, `src/dec/frame_dec.c`)
- `LoopFilter` - the edge filters, the filter-strength setup and which macroblocks have their inner edges filtered (`src/dsp/dec.c`, `src/dec/frame_dec.c`, `src/dec/vp8_dec.c`)
- `VP8TokenEncoder`, `VP8TokenDecoder` - `PutCoeffs` (`src/enc/frame_enc.c`) and `GetCoeffsFast` (`src/dec/vp8_dec.c`)
- `ChromaUpsampler` - the fancy chroma upsampler and the YUV-to-RGB conversion (`src/dsp/upsampling.c`, `src/dsp/yuv.h`, `src/dec/io_dec.c`)
- `VP8Encoder` - the rate-distortion lambdas and intra mode decision, `SetupMatrices`, `PickBestIntra16`, `PickBestIntra4` and `PickBestUV` (`src/enc/quant_enc.c`) with the flatness tests (`src/dsp/quant.h`), and the segment-map probabilities, `SetSegmentProbas` (`src/enc/frame_enc.c`)

Classes in `dev.simplified.image.codec.webp.vp8l`:

- `NearLosslessPreprocess` - near-lossless preprocessing (`src/enc/near_lossless_enc.c`)
- `LZ77`, `VP8LEncoder` - the plane-code tables and the code-length code order (`src/enc/backward_references_enc.c`)

```text
Copyright (c) 2010, Google Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### libvpx - The WebM Project authors

<https://chromium.googlesource.com/webm/libvpx>

Classes in `dev.simplified.image.codec.webp.vp8`:

- `NearMvs` - `vp8_find_near_mvs` (`vp8/common/findnearmv.c`)
- `SubpelPrediction` - the six-tap sub-pixel prediction filters (`vp8/common/filter.c`)
- `VP8Tables` - the inter-frame mode, motion-vector and SPLITMV tables (`vp8/common/entropymode.c`, `vp8/common/vp8_entropymodedata.h`, `vp8/common/entropymv.c`, `vp8/common/modecont.c`, `vp8/common/findnearmv.c`, `vp8/decoder/decodemv.c`)
- `VP8Encoder` - the default loop-filter deltas (`vp8/encoder/onyx_if.c`)

```text
Copyright (c) 2010, The WebM Project authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google, nor the WebM Project, nor the names
    of its contributors may be used to endorse or promote products
    derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### RFC 6386, VP8 Data Format and Decoding Guide - IETF Trust and the document authors

<https://www.rfc-editor.org/rfc/rfc6386>

`BooleanEncoder` in `dev.simplified.image.codec.webp.vp8` ports the boolean entropy encoder that section 7.3 of the RFC sets out.

```text
Copyright (c) 2011 IETF Trust and the persons identified as the
document authors. All rights reserved.

This document is subject to BCP 78 and the IETF Trust's Legal
Provisions Relating to IETF Documents
(http://trustee.ietf.org/license-info) in effect on the date of
publication of this document. Please review these documents
carefully, as they describe your rights and restrictions with respect
to this document.
```

RFC 6386 is an Independent Submission published in November 2011, so the Trust's Legal Provisions in effect then, version 4.0 of 28 December 2009, apply to it as a document of the Independent Submission Stream (section 8.f). For that stream they withhold the Simplified BSD License they grant the code in IETF-stream documents (section 4) and grant instead the licence of section 3.a, which section 8.c.iii frees of the IETF Standards Process: to copy, distribute, translate, and modify or prepare derivative works of the document, in whole or in part. The RFC carries no notice disallowing modification. The BSD licence in the RFC's section 20.26 covers the reference decoder source in its Attachment One, which nothing here ports.

## Upstream Libraries

| Library | License | Use |
|---------|---------|-----|
| [simplified-dev/collections](https://github.com/simplified-dev/collections) | Apache-2.0 | `Concurrent` and `ConcurrentList`, the lists frames and chunks are held in |
| [simplified-dev/utils](https://github.com/simplified-dev/utils) | Apache-2.0 | `ByteArrayDataOutput` the writers write through, and string and number helpers |
| [simplified-dev/annotations](https://github.com/simplified-dev/annotations) | Apache-2.0 | Annotation processor for accessors, builders and constructors (compile-only) |
| [JetBrains Annotations](https://github.com/JetBrains/java-annotations) | Apache-2.0 | Nullability and API-status annotations |
| [Log4j 2 API](https://logging.apache.org/log4j/2.x/) | Apache-2.0 | Logging API on the `api` configuration; no main source calls it |
| [JUnit 5](https://junit.org/junit5/) | EPL-2.0 | Test framework |
| [Hamcrest](https://hamcrest.org/JavaHamcrest/) | BSD-3-Clause | Test matchers |

The Vector API (`jdk.incubator.vector`) used by `PixelBuffer`'s SIMD kernels in `dev.simplified.image` is part of the OpenJDK and is governed by the JDK's own license.

The tests consult libwebp through Python's `webp` package and libvpx through an `ffmpeg` build when either is installed. Neither is a dependency of the library, and nothing of either is distributed with it.

## Attribution

- **Project author** - [CraftedFury](https://sbs.dev/)
- **Organization** - [Simplified-Dev](https://github.com/simplified-dev)
