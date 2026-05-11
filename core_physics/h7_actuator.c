#define _GNU_SOURCE
#include <unistd.h>
#include <sys/prctl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* I/O Priority Constants (if not defined in sys/syscall.h) */
#ifndef IOPRIO_WHO_PROCESS
#define IOPRIO_WHO_PROCESS 1
#endif

#ifndef IOPRIO_CLASS_SHIFT
#define IOPRIO_CLASS_SHIFT 13
#endif

#ifndef IOPRIO_PRIO_VALUE
#define IOPRIO_PRIO_VALUE(class, data) (((class) << IOPRIO_CLASS_SHIFT) | data)
#endif

/**
 * h7_set_cpu_affinity
 * Mueve un proceso a un conjunto específico de CPUs.
 */
int h7_set_cpu_affinity(pid_t pid, int cpu_mask) {
    cpu_set_t set;
    CPU_ZERO(&set);
    
    // Si la máscara es 1, usamos CPU 0; si es 2, CPU 1, etc.
    // Una implementación más compleja parsearía bits.
    for (int i = 0; i < 8; i++) {
        if (cpu_mask & (1 << i)) {
            CPU_SET(i, &set);
        }
    }
    
    if (sched_setaffinity(pid, sizeof(set), &set) == -1) {
        perror("❌ Error en h7_set_cpu_affinity");
        return -1;
    }
    return 0;
}

/**
 * h7_set_sched_policy
 * Cambia la política del scheduler (FIFO, RR, BATCH, NORMAL).
 */
int h7_set_sched_policy(pid_t pid, int policy, int priority) {
    struct sched_param param;
    param.sched_priority = priority;
    
    if (sched_setscheduler(pid, policy, &param) == -1) {
        perror("❌ Error en h7_set_sched_policy");
        return -1;
    }
    return 0;
}

/**
 * h7_set_io_priority
 * Ajusta la prioridad de entrada/salida de disco.
 */
int h7_set_io_priority(pid_t pid, int io_class, int io_priority) {
    // syscall(SYS_ioprio_set, which, who, ioprio)
    if (syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, pid, 
                IOPRIO_PRIO_VALUE(io_class, io_priority)) == -1) {
        perror("❌ Error en h7_set_io_priority");
        return -1;
    }
    return 0;
}
