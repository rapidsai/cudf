/*
 *
 *  Copyright (c) 2023, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

/**
 * Represents some amount of host memory that has been reserved. A reservation guarantees that one
 * or more allocations up to the reserved amount, minus padding for alignment will succeed. A
 * reservation typically guarantees the amount can be allocated one, meaning when a buffer
 * allocated from a reservation is freed it is not returned to the reservation, but to the pool of
 * memory the reservation originally came from. If more memory is allocated from the reservation
 * an OutOfMemoryError may be thrown, but it is not guaranteed to happen.
 *
 * When the reservation is closed any unused reservation will be returned to the pool of memory
 * the reservation came from.
 */
public interface HostMemoryReservation extends HostMemoryAllocator, AutoCloseable {}
