package server

import "fmt"

// Server handles requests.
type Server struct {
	Name string
}

// Start begins serving.
func (s *Server) Start() error {
	fmt.Println("starting", s.Name)
	return nil
}

func NewServer(name string) *Server {
	return &Server{Name: name}
}
