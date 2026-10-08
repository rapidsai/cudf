// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{AnalyzerError, AnalyzerResult};
use quent_dynamic_attributes::DynamicAttribute;
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition};
use uuid::Uuid;

use crate::{generated::MemoryReservationEvent, resource::MEMORY_RESERVATION_ENTITY_TYPE};

#[derive(Default)]
pub(crate) struct MemoryReservationBuilder {
    instance_name: Option<String>,
    actor_id: Option<Uuid>,
    requested_at: Option<TimeUnixNanoSec>,
    requested_attributes: Vec<DynamicAttribute>,
    finished_at: Option<TimeUnixNanoSec>,
    finished_state: Option<&'static str>,
    finished_attributes: Vec<DynamicAttribute>,
}

impl MemoryReservationBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &MemoryReservationEvent) {
        match event {
            MemoryReservationEvent::Requested {
                instance_name,
                actor,
                request,
                ..
            } => {
                self.instance_name = Some(instance_name.clone());
                self.actor_id = Some(actor.target);
                self.requested_at = Some(timestamp);
                self.requested_attributes = vec![
                    DynamicAttribute::string("purpose", request.purpose.clone()),
                    DynamicAttribute::u64("size_bytes", request.size_bytes),
                    DynamicAttribute::string("memory_type", request.memory_type.clone()),
                    DynamicAttribute::i64("net_memory_delta", request.net_memory_delta),
                ];
                if let Some(allow_overbooking) = request.allow_overbooking {
                    self.requested_attributes.push(DynamicAttribute::u8(
                        "allow_overbooking",
                        u8::from(allow_overbooking),
                    ));
                }
                if let Some(sequence_number) = request.sequence_number {
                    self.requested_attributes
                        .push(DynamicAttribute::u64("sequence_number", sequence_number));
                }
            }
            MemoryReservationEvent::Granted { .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("granted");
            }
            MemoryReservationEvent::Failed { error, .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("failed");
                self.finished_attributes = vec![DynamicAttribute::string("error", error.clone())];
            }
        }
    }

    pub(crate) fn try_build(self, id: Uuid) -> AnalyzerResult<MemoryReservationSpan> {
        let incomplete = |field| {
            AnalyzerError::IncompleteEntity(format!("memory reservation {id} is missing {field}"))
        };
        let instance_name = self
            .instance_name
            .ok_or_else(|| incomplete("instance name"))?;
        let actor_id = self.actor_id.ok_or_else(|| incomplete("actor"))?;
        let start = self
            .requested_at
            .ok_or_else(|| incomplete("requested event"))?;
        let end = self
            .finished_at
            .ok_or_else(|| incomplete("granted or failed event"))?;
        let finished_state = self
            .finished_state
            .ok_or_else(|| incomplete("finished state"))?;
        Ok(MemoryReservationSpan {
            id,
            instance_name,
            actor_id,
            span: SpanUnixNanoSec::try_new(start, end)?,
            requested_attributes: self.requested_attributes,
            finished_state,
            finished_attributes: self.finished_attributes,
        })
    }
}

pub(crate) struct MemoryReservationSpan {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) actor_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
    requested_attributes: Vec<DynamicAttribute>,
    finished_state: &'static str,
    finished_attributes: Vec<DynamicAttribute>,
}

impl MemoryReservationSpan {
    pub(crate) fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        let transition = |name: &str, timestamp, attributes| FsmTransition {
            name: name.to_owned(),
            usages: vec![],
            timestamp: to_secs_relative(timestamp, epoch),
            attributes,
            derived_attributes: vec![],
        };
        FiniteStateMachine {
            id: self.id,
            type_name: MEMORY_RESERVATION_ENTITY_TYPE.to_owned(),
            instance_name: self.instance_name.clone(),
            transitions: vec![
                transition(
                    "requested",
                    self.span.start(),
                    self.requested_attributes.clone(),
                ),
                transition(
                    self.finished_state,
                    self.span.end(),
                    self.finished_attributes.clone(),
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use quent_events::EntityRef;

    use super::*;

    #[test]
    fn builds_granted_reservation_wait() {
        let mut builder = MemoryReservationBuilder::default();
        builder.push(
            10,
            &MemoryReservationEvent::Requested {
                seq: 0,
                instance_name: "scan-reservation".to_owned(),
                actor: EntityRef::new(Uuid::now_v7(), ()),
                request: crate::generated::MemoryReservationRequest {
                    purpose: "scan".to_owned(),
                    size_bytes: 20,
                    memory_type: "DEVICE".to_owned(),
                    net_memory_delta: 10,
                    allow_overbooking: Some(false),
                    sequence_number: Some(2),
                },
            },
        );
        builder.push(30, &MemoryReservationEvent::Granted { seq: 1 });

        let reservation = builder.try_build(Uuid::now_v7()).unwrap();
        let fsm = reservation.to_ui_fsm(0);

        assert_eq!(reservation.span.duration(), 20);
        assert_eq!(fsm.transitions[0].name, "requested");
        assert_eq!(fsm.transitions[1].name, "granted");
        assert_eq!(
            fsm.transitions[0].attributes,
            vec![
                DynamicAttribute::string("purpose", "scan".to_owned()),
                DynamicAttribute::u64("size_bytes", 20),
                DynamicAttribute::string("memory_type", "DEVICE".to_owned()),
                DynamicAttribute::i64("net_memory_delta", 10),
                DynamicAttribute::u8("allow_overbooking", 0),
                DynamicAttribute::u64("sequence_number", 2),
            ]
        );
    }
}
