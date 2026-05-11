CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC
LDFLAGS = -shared -lm

TARGET_LIB = libmetriplex_core.so
TARGET_DAEMON = ../h7_daemon

SOURCES = metriplex_core.c
OBJECTS = $(SOURCES:.c=.o)

all: $(TARGET_LIB) $(TARGET_DAEMON)

$(TARGET_LIB): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(TARGET_DAEMON): ../h7_daemon.c $(OBJECTS)
	$(CC) $(CFLAGS) -I. $^ -o $@ -lm

%.o: %.c metriplex_core.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET_LIB) $(TARGET_DAEMON)
