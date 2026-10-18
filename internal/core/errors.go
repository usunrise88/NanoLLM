package core

import (
	"errors"
	"fmt"
	"net/http"
)

// Code is a stable, machine-readable error identifier. Dialects map it to their
// own wire format; the set itself never depends on a dialect.
type Code string

const (
	CodeInvalidRequest  Code = "invalid_request"
	CodeUnauthorized    Code = "unauthorized"
	CodeForbidden       Code = "forbidden"
	CodeModelNotFound   Code = "model_not_found"
	CodeJobNotFound     Code = "job_not_found"
	CodeContextOverflow Code = "context_overflow"
	// CodeDeviceUnavailable: the request cannot run on the device it asked for
	// and fallback is off (SPEC §3).
	CodeDeviceUnavailable Code = "device_unavailable"
	// CodeCapabilityUnavailable: the device or backend cannot do what was asked
	// — a grammar on the NPU, for one.
	CodeCapabilityUnavailable Code = "capability_unavailable"
	CodeQueueFull             Code = "queue_full"
	CodeRateLimited           Code = "rate_limited"
	CodeModelUnavailable      Code = "model_unavailable"
	CodeNotImplemented        Code = "not_implemented"
	CodeInternal              Code = "internal"
)

// Error is the single error type crossing the service boundary.
type Error struct {
	Code    Code   `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	cause   error
}

func (e *Error) Error() string {
	out := fmt.Sprintf("%s: %s", e.Code, e.Message)
	if e.Param != "" {
		out += fmt.Sprintf(" (param %s)", e.Param)
	}
	// The cause appears in the Go error string, which the CLI prints and the
	// logger records. API responses render Message on its own, so nothing
	// internal reaches a client through this.
	if e.cause != nil {
		out += ": " + e.cause.Error()
	}
	return out
}

func (e *Error) Unwrap() error { return e.cause }

// Errorf builds a domain error.
func Errorf(code Code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// WithParam names the offending request field.
func (e *Error) WithParam(p string) *Error { e.Param = p; return e }

// WithCause attaches an internal cause. The cause is logged, never returned to
// a client.
func (e *Error) WithCause(err error) *Error { e.cause = err; return e }

// AsError returns err as a domain error, wrapping anything else as internal so
// that an unexpected failure never leaks its text to a client.
func AsError(err error) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	return Errorf(CodeInternal, "internal error").WithCause(err)
}

// HTTPStatus is the status a dialect should use unless it has a reason not to.
func (c Code) HTTPStatus() int {
	switch c {
	case CodeInvalidRequest, CodeContextOverflow:
		return http.StatusBadRequest
	case CodeUnauthorized:
		return http.StatusUnauthorized
	case CodeForbidden:
		return http.StatusForbidden
	case CodeModelNotFound, CodeJobNotFound:
		return http.StatusNotFound
	case CodeDeviceUnavailable, CodeCapabilityUnavailable:
		return http.StatusConflict
	case CodeQueueFull, CodeRateLimited:
		return http.StatusTooManyRequests
	case CodeModelUnavailable:
		return http.StatusServiceUnavailable
	case CodeNotImplemented:
		return http.StatusNotImplemented
	default:
		return http.StatusInternalServerError
	}
}
