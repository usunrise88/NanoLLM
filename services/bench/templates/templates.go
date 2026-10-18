// Package templates embeds the pages (html/template) and their static
// files. No build step: what is here is what is served.
package templates

import "embed"

//go:embed *.html static
var FS embed.FS
