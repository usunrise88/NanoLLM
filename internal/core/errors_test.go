package core

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
)

func TestAsErrorCancellation(t *testing.T) {
	for _, err := range []error{context.Canceled, fmt.Errorf("prefill: %w", context.Canceled)} {
		e := AsError(err)
		if e.Code != CodeCancelled || e.Code.HTTPStatus() != StatusClientClosedRequest {
			t.Errorf("%v: %s %d", err, e.Code, e.Code.HTTPStatus())
		}
		if !errors.Is(e, context.Canceled) {
			t.Errorf("%v: the cause is lost", err)
		}
	}
	// Anything else unexpected stays an internal error; a deadline is not a hang-up.
	for _, err := range []error{errors.New("boom"), context.DeadlineExceeded} {
		if e := AsError(err); e.Code != CodeInternal || e.Code.HTTPStatus() != http.StatusInternalServerError {
			t.Errorf("%v: %s", err, e.Code)
		}
	}
	// A domain error keeps its own code even when it wraps a cancellation.
	if e := AsError(Errorf(CodeQueueFull, "full").WithCause(context.Canceled)); e.Code != CodeQueueFull {
		t.Errorf("domain error rewritten: %s", e.Code)
	}
}
