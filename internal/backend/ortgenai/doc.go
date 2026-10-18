// Package ortgenai is the optional NPU backend (SPEC §3, v2): onnxruntime-genai
// through purego, with the QNN (Qualcomm), VitisAI (AMD) and OpenVINO (Intel)
// execution providers.
//
// It is loaded only when the library is present in lib/ or on the system, is
// never picked by device "auto" while the model is unverified on the NPU, and
// does not support grammars: such requests fail with 409 or fall back.
//
// Empty until v2.
package ortgenai
