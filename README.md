# micrograd

It's a Golang port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) following his tutorial [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) 
It's made for educational purposes only. 
Do not use in production. 
Feedback is welcome.  

## Development

```bash
go test -race -coverprofile=cover.out -count=2 ./...
```

