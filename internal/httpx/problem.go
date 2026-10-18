// Package httpx holds the HTTP plumbing every dialect shares: the problem
// response, authentication and middleware.
package httpx

import (
	"encoding/json"
	"net/http"

	"github.com/usunrise88/nanollm/internal/core"
)

// Problem is an RFC 9457 problem details object, extended with the stable
// error code so a client can branch without parsing prose.
type Problem struct {
	Type   string `json:"type"`
	Title  string `json:"title"`
	Status int    `json:"status"`
	Detail string `json:"detail,omitempty"`
	Code   string `json:"code"`
	Param  string `json:"param,omitempty"`
}

// WriteProblem renders err as application/problem+json. Errors that are not
// domain errors are reported as internal without their text.
func WriteProblem(w http.ResponseWriter, err error) {
	p := ProblemOf(err)
	w.Header().Set("Content-Type", "application/problem+json")
	w.WriteHeader(p.Status)
	_ = json.NewEncoder(w).Encode(p)
}

// ProblemOf is err as a problem object, for an error event on a stream that
// has already started.
func ProblemOf(err error) Problem {
	e := core.AsError(err)
	status := e.Code.HTTPStatus()
	return Problem{
		Type:   "about:blank",
		Title:  http.StatusText(status),
		Status: status,
		Detail: e.Message,
		Code:   string(e.Code),
		Param:  e.Param,
	}
}

// WriteJSON writes v with status.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// NotYet is the error for a route that is part of the API but not of this
// build yet. It names the milestone, so a client author reading the response
// knows whether to wait or to look for a mistake.
func NotYet(what, milestone string) error {
	return core.Errorf(core.CodeNotImplemented, "%s lands in %s", what, milestone)
}
