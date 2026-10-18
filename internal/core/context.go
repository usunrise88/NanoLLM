package core

import "context"

type interactiveKey struct{}

// WithInteractive marks a request as interactive: it overtakes batch work in
// the scheduler. The HTTP layer sets it from the API key, never from a request
// field (see config.APIKey.Interactive).
func WithInteractive(ctx context.Context, interactive bool) context.Context {
	return context.WithValue(ctx, interactiveKey{}, interactive)
}

// Interactive reports whether the request was marked interactive.
func Interactive(ctx context.Context) bool {
	v, _ := ctx.Value(interactiveKey{}).(bool)
	return v
}
