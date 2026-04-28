CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC
LDFLAGS = -shared -lm

TARGET = libmetriplex_core.so
SOURCES = metriplex_core.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c metriplex_core.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
